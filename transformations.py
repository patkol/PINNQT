# Copyright (c) 2024 ETH Zurich, Patrice Kolb

# OPTIM: don't return qs in trafos


from typing import Dict, Callable, Sequence, Optional
import copy
import itertools
import numpy as np
import scipy.interpolate  # type: ignore
import torch

from kolpinn import mathematics
from kolpinn.mathematics import complex_abs2
from kolpinn import grids
from kolpinn.grids import Grid, Subgrid, Supergrid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import model

import physical_constants as consts
import parameters as params
import physics
from classes import Contact


def transition_f(x: torch.Tensor):
    """
    Smooth transition from 0 at x <= 0 to 1 at x -> inf
    https://en.wikipedia.org/wiki/Bump_function#Examples
    """
    # Assert that we don't ignore nonfinite x due to the nan_to_num
    assert torch.all(torch.isfinite(x)), x

    exp = torch.exp(-1 / x)
    clean_exp = torch.nan_to_num(exp)  # 0 at x=0

    return clean_exp * (x > 0)


def transition_g(x: torch.Tensor):
    """
    Smooth transition from 0 at x <= 0 to 1 at x >= 1
    https://en.wikipedia.org/wiki/Bump_function#Examples
    """
    f_x = transition_f(x)
    return f_x / (f_x + transition_f(1 - x))


def smooth_transition(x: torch.Tensor, x0: float, x1: float, y0: float, y1: float):
    """
    Smooth transition from y0 at x0 to y1 at x1
    """
    g = transition_g((x - x0) / (x1 - x0))
    return (y1 - y0) * g + y0


def k_function(q: QuantityDict, i: int, contact: Contact) -> torch.Tensor:
    return physics.k_function(
        q[f"m_eff{i}"],
        q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el{i}"],
    )


def smoothen(quantity: torch.Tensor, grid: Grid, dim_label: str):
    """
    Smoothen `quantity` on `grid` along `dim_label`.
    `grid` must be sorted along `dim_label`.
    """

    assert quantities.compatible(quantity, grid)

    other_dim_labels = copy.copy(grid.dimensions_labels)
    other_dim_labels.remove(dim_label)
    other_ranges = [
        range(quantity.size(grid.index[other_dim_label]))
        for other_dim_label in other_dim_labels
    ]

    smoothened_quantity = torch.zeros_like(quantity)

    # scipy.interpolate.UnivariateSpline only works with 1D data, so we have to
    # call it for all combinations of the other dimensions.
    for other_indices in itertools.product(*other_ranges):
        # Get current slice
        slices = [slice(None)] * grid.n_dim
        for other_dim_label, other_index in zip(other_dim_labels, other_indices):
            slices[grid.index[other_dim_label]] = other_index

        # Smoothen slice
        real_spline = scipy.interpolate.UnivariateSpline(
            grid[dim_label].cpu(),
            torch.real(quantity[slices]).cpu(),
            s=1,  # TODO: make adjustable
        )
        imag_spline = scipy.interpolate.UnivariateSpline(
            grid[dim_label].cpu(),
            torch.imag(quantity[slices]).cpu(),
            s=1,  # TODO: make adjustable
        )
        smoothened_quantity[slices] = torch.from_numpy(
            real_spline(grid[dim_label].cpu()) + 1j * imag_spline(grid[dim_label].cpu())
        )

    return smoothened_quantity


def get_V_voltage(q: QuantityDict, device_start, device_end):
    distance_factor = (q["x"] - device_start) / (device_end - device_start)
    # Cap the factor beyond the device limits, these regions correspond to
    # the contacts
    distance_factor[distance_factor < 0] = 0
    distance_factor[distance_factor > 1] = 1
    V_voltage = -q["voltage"] * consts.EV * distance_factor

    return V_voltage


def get_dx_model(mode: str, quantity_name: str, grid_name: str):
    name = quantity_name + "_dx"
    if mode == "exact":
        dx_model_single = model.FunctionModel(
            lambda q, *, q_full, with_grad: mathematics.grad(
                q[quantity_name],
                q_full["x"],
                retain_graph=True,  # OPTIM: not always necessary
                create_graph=True,  # OPTIM: not always necessary
            ),
            q_full=None,
            with_grad=True,
        )

        return model.get_multi_model(dx_model_single, name, grid_name)

    if mode == "multigrid":

        def dx_qs_trafo(qs):
            quantity_right = qs[grid_name + "_pdx"][quantity_name]
            quantity_left = qs[grid_name + "_mdx"][quantity_name]
            qs[grid_name][name] = (quantity_right - quantity_left) / (2 * params.dx)
            return qs

    elif mode == "singlegrid":

        def dx_qs_trafo(qs):
            q = qs[grid_name]
            q[name] = quantities.get_fd_derivative("x", q[quantity_name], q.grid)
            return qs

    else:
        raise ValueError("Unknown dx mode: ", mode)

    return model.MultiModel(dx_qs_trafo, name)


def get_E_fermi(q: QuantityDict, *, i: int):
    """The potential V still has to be added."""
    if params.CONSTANT_FERMI_LEVEL is not None:
        return params.CONSTANT_FERMI_LEVEL

    dos = 8 * np.pi / consts.H**3 * torch.sqrt(2 * q[f"m_eff{i}"] ** 3 * q["DeltaE"])

    best_E_f = None
    best_abs_remaining_charge = float("inf")
    E_fs = np.arange(0 * consts.EV, 0.8 * consts.EV, 1e-3 * consts.EV)
    for E_f in E_fs:
        fermi_dirac = 1 / (1 + torch.exp(physics.BETA * (q["DeltaE"] - E_f)))
        integrand = dos * fermi_dirac
        # Particle, not charge density
        n = quantities.sum_dimension("DeltaE", integrand, q.grid) * params.E_STEP
        abs_remaining_charge = torch.abs(n - q[f"doping{i}"]).item()
        if abs_remaining_charge < best_abs_remaining_charge:
            best_abs_remaining_charge = abs_remaining_charge
            best_E_f = E_f

        E_f += 1e-3 * consts.EV

    assert best_E_f is not None
    assert best_E_f != E_fs[0] and best_E_f != E_fs[-1], E_f / consts.EV

    relative_remaining_charge = best_abs_remaining_charge / q[f"doping{i}"]
    print(
        f"Best fermi energy: {best_E_f / consts.EV} eV with a relative remaining charge of {relative_remaining_charge}"
    )

    return best_E_f


def get_phi_target(
    q: QuantityDict,
    *,
    i: int,
    contact: Contact,
    direction: int,
    get_derivative: bool = True,
):
    """
    Boundary conditions: Given phi{i_prev}, find phi{i} at the boundary between
    i and i_prev.
    i_prev is towards the input contact if direction == 1 and towards the output
    if direction == -1.
    """
    assert direction in (1, -1)

    i_prev = i - contact.direction * direction
    phi_target = torch.clone(q[f"phi{i_prev}_{contact}"])
    if get_derivative:
        phi_dx_target = (
            q[f"m_eff{i}"] / q[f"m_eff{i_prev}"] * q[f"phi{i_prev}_{contact}_dx"]
        )
    else:
        phi_dx_target = None

    return phi_target, phi_dx_target


def get_fermi_integral(*, m_eff, E_fermi, E):
    return (
        m_eff
        / (np.pi * consts.H_BAR**2 * physics.BETA)
        * torch.log(1 + torch.exp(physics.BETA * (E_fermi - E)))
    )


def get_n_contact(*, dos, fermi_integral, grid: Grid):
    integrand = dos * fermi_integral
    n_contact = quantities.sum_dimension("DeltaE", integrand, grid) * params.E_STEP
    return n_contact


def get_simple_phi(q: QuantityDict, *, i: int, contact: Contact):
    """Adapt get_simple_phi_dx when changing this!"""
    phi = torch.clone(q[f"phi_zero{i}_{contact}"])
    if params.use_phi_one:
        phi += q[f"phi_one{i}_{contact}"]

    return phi


def get_simple_phi_dx(q: QuantityDict, *, i: int, contact: Contact):
    """Dependent on get_simple_phi - not ideal, might be solved with lazy eval"""
    phi_dx = torch.clone(q[f"phi_zero{i}_{contact}_dx"])
    if params.use_phi_one:
        phi_dx += q[f"phi_one{i}_{contact}_dx"]

    return phi_dx


def to_full_trafo(
    qs: Dict[str, QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int], str],
    quantity_label: str,
):
    """
    Combine quantities in the 'bulk{i}' grids to 'bulk'.
    They're called 'label_fn(i)' in the bulk grids and
    'quantity_label' in 'bulk'.
    """
    quantity_list: list[torch.Tensor] = []
    subgrid_list: list[Subgrid] = []
    for i in range(1, N + 1):
        q_layer = qs[f"bulk{i}"]
        quantity_list.append(q_layer[label_fn(i)])
        assert isinstance(q_layer.grid, Subgrid)
        subgrid_list.append(q_layer.grid)

    qs["bulk"][quantity_label] = quantities.combine_quantity(
        quantity_list,
        subgrid_list,
        qs["bulk"].grid,
    )

    return qs


def to_bulks_trafo(
    qs: dict[str, QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int], str],
    quantity_label: str,
):
    """
    Inverse of `to_full_trafo`
    """
    full_quantity = qs["bulk"][quantity_label]
    for i in range(1, N + 1):
        q_layer = qs[f"bulk{i}"]
        assert isinstance(q_layer.grid, Subgrid)
        q_layer[label_fn(i)] = quantities.restrict(full_quantity, q_layer.grid)

    return qs


def _interpolating_to_boundaries_trafo(
    qs: Dict[str, QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int], str],
    quantity_label: str,
    dx_dict: Dict[str, float],
):
    full_quantity = qs["bulk"][quantity_label]
    full_grid = qs["bulk"].grid
    for i in range(0, N + 1):
        for dx_string in dx_dict.keys():
            q = qs[f"boundary{i}" + dx_string]
            boundary_quantity = quantities.interpolate(
                full_quantity,
                full_grid,
                q.grid,
                dimension_label="x",
            )
            q[label_fn(i)] = boundary_quantity
            q[label_fn(i + 1)] = boundary_quantity

    return qs


def _extrapolating_to_boundaries_trafo(
    qs: Dict[str, QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int], str],
    dx_dict: Dict[str, float],
):
    for i_layer in range(1, N + 1):
        q_bulk = qs[f"bulk{i_layer}"]
        label = label_fn(i_layer)
        for i_boundary in (i_layer - 1, i_layer):
            for dx_string in dx_dict.keys():
                q_boundary = qs[f"boundary{i_boundary}" + dx_string]
                q_boundary[label] = quantities.interpolate(
                    q_bulk[label],
                    q_bulk.grid,
                    q_boundary.grid,
                    dimension_label="x",
                )

    return qs


def to_boundaries_trafo(
    qs: Dict[str, QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int], str],
    quantity_label: Optional[str] = None,
    dx_dict: Dict[str, float],
    one_sided: bool,
):
    """
    Interpolate from "bulk" to the boundaries.
    If `one_sided`, we extrapolate from the "bulk{i}" individually, leading to possibly
    different values for i and i+1 at the same boundary.
    """

    if one_sided:
        return _extrapolating_to_boundaries_trafo(
            qs, N=N, label_fn=label_fn, dx_dict=dx_dict
        )

    assert quantity_label is not None

    return _interpolating_to_boundaries_trafo(
        qs, N=N, label_fn=label_fn, quantity_label=quantity_label, dx_dict=dx_dict
    )


def to_bulks_and_boundaries_trafo(
    qs: dict[str, QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int], str],
    quantity_label: str,
    dx_dict: Dict[str, float],
    one_sided: bool,
):
    to_bulks_trafo(qs, N=N, label_fn=label_fn, quantity_label=quantity_label)
    to_boundaries_trafo(
        qs,
        N=N,
        label_fn=label_fn,
        quantity_label=quantity_label,
        dx_dict=dx_dict,
        one_sided=one_sided,
    )

    return qs


def wkb_phase_trafo(
    qs: dict[str, QuantityDict],
    *,
    contact: Contact,
    N: int,
    dx_dict: Dict[str, float],
):
    for i in range(1, N + 1):
        # Set up the grid to integrate on
        grid_names = [f"bulk{i}"]
        for i_boundary in (i - 1, i):
            for dx_string in dx_dict.keys():
                grid_names.append(f"boundary{i_boundary}" + dx_string)
        child_grids: dict[str, Grid] = dict(
            (grid_name, qs[grid_name].grid) for grid_name in grid_names
        )
        supergrid = Supergrid(child_grids, "x", copy_all=False)
        sorted_supergrid = grids.get_sorted_grid_along(["x"], supergrid, copy_all=False)
        ks = [qs[grid_name][f"k{i}_{contact}"] for grid_name in grid_names]
        integrand = quantities.combine_quantity(
            ks, list(supergrid.subgrids.values()), supergrid
        )
        sorted_integrand = quantities.restrict(integrand, sorted_supergrid)

        left_boundary_index = i - 1
        right_boundary_index = i
        x_L = qs[f"boundary{left_boundary_index}"].grid["x"].item()

        sorted_k_integral = quantities.get_cumulative_integral(
            "x", x_L, sorted_integrand, sorted_supergrid
        )
        sorted_k_integral = smoothen(sorted_k_integral, sorted_supergrid, "x")

        # Propagation left -> right
        sorted_LR_phase = torch.exp(1j * sorted_k_integral)
        LR_phase = quantities.combine_quantity(
            [sorted_LR_phase], [sorted_supergrid], supergrid
        )

        # Propagation right -> left
        right_k_integral = quantities.restrict(
            sorted_k_integral, supergrid.subgrids[f"boundary{right_boundary_index}"]
        )
        sorted_RL_phase = torch.exp(-1j * (sorted_k_integral - right_k_integral))
        RL_phase = quantities.combine_quantity(
            [sorted_RL_phase], [sorted_supergrid], supergrid
        )

        # Make a the in and b the out direction
        if contact.direction == 1:
            a_phase, b_phase = LR_phase, RL_phase
        else:
            a_phase, b_phase = RL_phase, LR_phase

        if params.ansatz == "half_wkb":
            if params.hard_bc_dir == -1:
                a_phase = torch.ones_like(a_phase)
            else:
                b_phase = torch.ones_like(b_phase)

        for grid_name in grid_names:
            q = qs[grid_name]
            q[f"a_phase{i}_{contact}"] = quantities.restrict(
                a_phase, supergrid.subgrids[grid_name]
            )
            if not params.use_phi_one:
                continue
            q[f"b_phase{i}_{contact}"] = quantities.restrict(
                b_phase, supergrid.subgrids[grid_name]
            )

    return qs


def E_fermi_trafo(qs, *, contact: Contact):
    q_in = qs[contact.grid_name]
    i = contact.index
    qs["bulk"][f"E_fermi_{contact}"] = (
        get_E_fermi(q_in, i=i) + q_in[f"V_int{i}"] + q_in[f"V_el{i}"]
    )

    return qs


def fermi_integral_trafo(qs, *, contact: Contact):
    q = qs["bulk"]
    q_in = qs[contact.grid_name]
    i = contact.index
    qs["bulk"][f"fermi_integral_{contact}"] = get_fermi_integral(
        m_eff=q_in[f"m_eff{i}"], E_fermi=q[f"E_fermi_{contact}"], E=q[f"E_{contact}"]
    )

    return qs


def phi_zero_one_trafo(qs, *, i: int, contact: Contact, grid_names: Sequence[str]):
    """
    Add phi_zero and phi_one.
    """

    for grid_name in grid_names:
        q = qs[grid_name]

        q[f"phi_zero{i}_{contact}"] = (
            q[f"a_output{i}_{contact}"] * q[f"a_phase{i}_{contact}"]
        )

        if not params.use_phi_one:
            continue

        q[f"phi_one{i}_{contact}"] = (
            q[f"b_output{i}_{contact}"] * q[f"b_phase{i}_{contact}"]
        )

    return qs


def simple_phi_trafo(qs, *, i: int, contact: Contact, grid_names: Sequence[str]):
    """phi0 + phi1 (BC not forced)"""

    for grid_name in grid_names:
        q = qs[grid_name]
        q[f"phi{i}_{contact}"] = get_simple_phi(q, i=i, contact=contact)

    return qs


def hard_bc_phi_trafo_conj(
    qs, *, i: int, contact: Contact, grid_names: Sequence[str], direction: int
):
    """d0 phi0 + d1 conj(phi0)"""

    assert direction in (1, -1)

    boundary_index = (
        contact.get_in_boundary_index(i)
        if direction == 1
        else contact.get_out_boundary_index(i)
    )
    q_boundary = qs[f"boundary{boundary_index}"]

    phi_zero = q_boundary[f"phi_zero{i}_{contact}"]
    phi_zero_dx = q_boundary[f"phi_zero{i}_{contact}_dx"]
    phi_target, phi_dx_target = get_phi_target(
        q_boundary, i=i, contact=contact, direction=direction
    )

    # phi = u * phi_zero + v * conj(phi_zero), find u & v s.t. this matches the target
    determinant = 2j * torch.imag(phi_zero * torch.conj(phi_zero_dx))
    u = (
        phi_target * torch.conj(phi_zero_dx) - torch.conj(phi_zero) * phi_dx_target
    ) / determinant
    v = -(phi_target * phi_zero_dx - phi_zero * phi_dx_target) / determinant

    for grid_name in grid_names:
        q = qs[grid_name]
        phi_zero_full = q[f"phi_zero{i}_{contact}"]
        q[f"phi{i}_{contact}"] = u * phi_zero_full + v * torch.conj(phi_zero_full)

    return qs


def hard_bc_phi_trafo_with_phi_one(
    qs, *, i: int, contact: Contact, grid_names: Sequence[str], direction: int
):
    """d0 phi0 + d1 phi1"""

    assert direction in (1, -1)

    boundary_index = (
        contact.get_in_boundary_index(i)
        if direction == 1
        else contact.get_out_boundary_index(i)
    )
    q_boundary = qs[f"boundary{boundary_index}"]

    phi_zero = q_boundary[f"phi_zero{i}_{contact}"]
    phi_zero_dx = q_boundary[f"phi_zero{i}_{contact}_dx"]
    phi_one = q_boundary[f"phi_one{i}_{contact}"]
    phi_one_dx = q_boundary[f"phi_one{i}_{contact}_dx"]
    phi_target, phi_dx_target = get_phi_target(
        q_boundary, i=i, contact=contact, direction=direction
    )

    determinant = phi_zero * phi_one_dx - phi_zero_dx * phi_one
    d_zero = (phi_one_dx * phi_target - phi_one * phi_dx_target) / determinant
    d_one = (-phi_zero_dx * phi_target + phi_zero * phi_dx_target) / determinant

    for grid_name in grid_names:
        q = qs[grid_name]
        phi_zero_full = q[f"phi_zero{i}_{contact}"]
        phi_one_full = q[f"phi_one{i}_{contact}"]
        q[f"phi{i}_{contact}"] = d_zero * phi_zero_full + d_one * phi_one_full

    return qs


def hard_bc_in_phi_trafo(
    qs, *, i: int, contact: Contact, grid_names: Sequence[str], direction: int
):
    """
    d * (phi0 [+ phi1]) at the input boundary.
    Condition: phi' = gamma * (2 - phi) at the input contact.
    with gamma := m_eff_device / m_eff_contact * i * k_contact.
    Solved by d * phi_old
    with d = 2 / (phi_old_contact' / gamma + phi_old_contact)
    """

    assert direction == 1  # Don't need this special trafo for output -> input
    assert i == contact.in_layer_index

    boundary_index = contact.in_boundary_index
    q_boundary = qs[f"boundary{boundary_index}"]

    phi_boundary = get_simple_phi(q_boundary, i=i, contact=contact)
    phi_dx_boundary = get_simple_phi_dx(q_boundary, i=i, contact=contact)
    gamma = (
        q_boundary[f"m_eff{i}"]
        / q_boundary[f"m_eff{contact.index}"]
        * 1j
        * q_boundary[f"k{contact.index}_{contact}"]
    )
    d = 2 / (phi_dx_boundary / gamma + phi_boundary)

    for grid_name in grid_names:
        q = qs[grid_name]
        phi = get_simple_phi(q, i=i, contact=contact)
        q[f"phi{i}_{contact}"] = d * phi

    return qs


def hard_bc_out_phi_trafo(
    qs, *, i: int, contact: Contact, grid_names: Sequence[str], direction: int
):
    """
    Force the output BC at the output boundary without affecting the WF
    at the input boundary.
    BC on the output boundary: There can only be an
    outgoing wave there. The criterion for that is i*k*phi = phi_dx.
    Assuming that there the eff mass & potential plateau towards the
    contacts - then the solution should approach e^{ikx}.
    """

    assert direction == 1  # Don't need this special trafo for output -> input
    assert i == contact.out_layer_index

    q_in_boundary = qs[f"boundary{contact.get_in_boundary_index(i)}"]
    q_out_boundary = qs[f"boundary{contact.get_out_boundary_index(i)}"]
    x_in, x_out = q_in_boundary["x"].item(), q_out_boundary["x"].item()

    for grid_name in grid_names:
        q = qs[grid_name]
        delta_x = q["x"] - q_out_boundary["x"]
        plane_wave_phase = torch.exp(1j * q_out_boundary[f"k{i}_{contact}"] * delta_x)
        # TODO: Force the correct transmitted amplitude based on the reflected one
        phi_out = q_out_boundary[f"phi{i}_{contact}"] * plane_wave_phase

        # Transition to phi_out
        transition_function = smooth_transition(q["x"], x_in, x_out, 1, 0)
        q.overwrite(
            f"phi{i}_{contact}",
            transition_function * q[f"phi{i}_{contact}"]
            + (1 - transition_function) * phi_out,
        )

    return qs


def phi_trafo(qs, **kwargs):
    if params.hard_bc_dir == 0:
        return simple_phi_trafo(qs, **kwargs)

    # Input layer
    if params.hard_bc_dir == 1 and kwargs["i"] == kwargs["contact"].in_layer_index:
        hard_bc_in_phi_trafo(qs, **kwargs, direction=params.hard_bc_dir)

    # Non-input layers
    else:
        if params.use_phi_one:
            hard_bc_phi_trafo_with_phi_one(qs, **kwargs, direction=params.hard_bc_dir)
        else:
            hard_bc_phi_trafo_conj(qs, **kwargs, direction=params.hard_bc_dir)

    # Output layer
    if (
        params.hard_bc_output
        and params.hard_bc_dir == 1
        and kwargs["i"] == kwargs["contact"].out_layer_index
    ):
        hard_bc_out_phi_trafo(qs, **kwargs, direction=params.hard_bc_dir)

    return qs


def contact_coeffs_trafo(qs, *, contact):
    q_in = qs[contact.grid_name]
    q = qs["bulk"]
    q_in = qs[f"boundary{contact.in_boundary_index}"]
    q_out = qs[f"boundary{contact.out_boundary_index}"]

    # phi_in, phi_dx_in: The contact's phi and its derivative at the input boundary
    phi_in, phi_dx_in = get_phi_target(
        q_in,
        i=contact.index,
        contact=contact,
        direction=-1,
        get_derivative=params.hard_bc_dir == -1,
    )

    if params.hard_bc_dir == -1:
        # transmitted_coeff is fixed
        """
        phi = a * exp(sik(x-x0)) + r * exp(-sik(x-x0)) with s = contact.direction,
        phi' = sik * (a * exp(sik(x-x0)) - r * exp(-sik(x-x0)))
        => a * exp(sik(x-x0)) = (phi + phi'/sik) / 2
        """
        k_in = q_in[f"k{contact.index}_{contact}"]
        q[f"incoming_coeff_{contact}"] = 0.5 * (
            phi_in + phi_dx_in / (contact.direction * 1j * k_in)
        )

    else:
        # incoming_coeff is fixed
        phi_out, _ = get_phi_target(
            q_out,
            i=contact.out_index,
            contact=contact,
            direction=1,
            get_derivative=False,
        )
        q[f"transmitted_coeff_{contact}"] = phi_out

    # incoming_coeff + reflected_coeff = phi_in
    q[f"reflected_coeff_{contact}"] = phi_in - q[f"incoming_coeff_{contact}"]

    return qs


def TR_trafo(qs, *, contact):
    """Transmission probability"""
    q = qs["bulk"]
    q_in = qs[contact.grid_name]
    q_out = qs[f"boundary{contact.out_boundary_index}"]
    incoming_amplitude = complex_abs2(q[f"incoming_coeff_{contact}"])
    reflected_amplitude = complex_abs2(q[f"reflected_coeff_{contact}"])
    transmitted_amplitude = complex_abs2(q[f"transmitted_coeff_{contact}"])
    real_v_in = torch.real(q_in[f"v{contact.index}_{contact}"])
    real_v_out = torch.real(q_out[f"v{contact.out_index}_{contact}"])
    T = transmitted_amplitude / incoming_amplitude * real_v_out / real_v_in
    R = reflected_amplitude / incoming_amplitude
    q[f"T_{contact}"] = T
    q[f"R_{contact}"] = R

    return qs


def I_contact_trafo(qs, *, contact):
    q = qs["bulk"]
    integrand = q[f"T_{contact}"] * q[f"fermi_integral_{contact}"]
    integral = quantities.sum_dimension("DeltaE", integrand, q.grid) * params.E_STEP
    prefactor = -consts.Q_E / consts.H_BAR / (2 * np.pi) * contact.direction
    q[f"I_spectrum_{contact}"] = prefactor * integrand
    q[f"I_{contact}"] = prefactor * integral

    return qs


def I_trafo(qs, *, contacts):
    q = qs["bulk"]
    q["I"] = sum(q[f"I_{contact}"] for contact in contacts)

    return qs


def j_exact_trafo(qs, *, contact):
    """
    Calculate the exact current at the output contact
    """
    q = qs[f"boundary{contact.out_boundary_index}"]
    k = q[f"k{contact.out_index}_{contact}"]
    m_eff = q[f"m_eff{contact.out_index}"]
    q[f"j_exact_{contact}"] = -contact.direction * consts.H_BAR * k / m_eff

    return qs


def dos_trafo(qs, *, contact):
    q = qs["bulk"]
    q_in = qs[contact.grid_name]

    incoming_amplitude = complex_abs2(q[f"incoming_coeff_{contact}"])
    v = q_in[f"v{contact.index}_{contact}"]
    # assert torch.allclose(
    #     torch.imag(v), torch.tensor(0, dtype=params.si_real_dtype)
    # ), "The energy in the input contact should be positive"
    dE_dk = consts.H_BAR * torch.real(v)
    q[f"DOS_{contact}"] = (
        1 / (2 * np.pi) * complex_abs2(q[f"phi_{contact}"]) / dE_dk / incoming_amplitude
    )
    return qs


def n_contact_trafo(qs, *, contact):
    """Density from one contact only"""
    q = qs["bulk"]
    q[f"n_{contact}"] = get_n_contact(
        dos=q[f"DOS_{contact}"],
        fermi_integral=q[f"fermi_integral_{contact}"],
        grid=q.grid,
    )
    return qs


def n_trafo(qs, *, contacts):
    """Full density"""
    q = qs["bulk"]
    q["n"] = sum(q[f"n_{contact}"] for contact in contacts)

    return qs


def V_electrostatic_trafo(qs, *, contacts, N: int):
    q = qs["bulk"]

    assert q.grid.dimensions_labels[-1] == "x"

    # Construct the discretized Laplace operator M assuming an
    # equispaced x grid
    # OPTIM: do this only once
    Nx = q.grid.dim_size["x"]
    dx = params.X_STEP
    M = torch.zeros(Nx, Nx)
    permittivity = quantities.squeeze_to(["x"], q["permittivity"], q.grid)
    for i in range(1, Nx):
        M[i, i - 1] = (permittivity[i] + permittivity[i - 1]) / (2 * dx**2)
    for i in range(0, Nx - 1):
        M[i, i + 1] = (permittivity[i] + permittivity[i + 1]) / (2 * dx**2)
    for i in range(1, Nx - 1):
        M[i, i] = -M[i, i - 1] - M[i, i + 1]

    # Von Neumann BC
    M[0, 0] = -(permittivity[0] + permittivity[1]) / (2 * dx**2)
    M[-1, -1] = -(permittivity[-1] + permittivity[-2]) / (2 * dx**2)

    # TODO: implementation that works if x is not the last coordinate,
    #       kolpinn function
    #       Then remove the assertion above
    rho = consts.Q_E * (q["doping"] - q["n"])
    Phi = q["V_el"] / -consts.Q_E
    F = torch.einsum("ij,...j->...i", M, Phi) + rho

    # Get the density if the potential was shifted by dV
    n_pdVs = []
    for contact in contacts:
        q_in = qs[contact.grid_name]
        i = contact.index
        fermi_integral_pdV = get_fermi_integral(
            m_eff=q_in[f"m_eff{i}"],
            E_fermi=q[f"E_fermi_{contact}"],
            E=q[f"E_{contact}"] + params.dV_poisson,
        )
        n_pdV = get_n_contact(
            dos=q[f"DOS_{contact}"],
            fermi_integral=fermi_integral_pdV,
            grid=q.grid,
        )
        n_pdVs.append(n_pdV)
    n_pdV = sum(n_pdVs)

    dn_dV = (n_pdV - q["n"]) / params.dV_poisson
    drho_dV = -consts.Q_E * dn_dV
    drho_dPhi = -consts.Q_E * drho_dV
    torch.unsqueeze(M, 0)
    torch.unsqueeze(M, 0)
    J = M + torch.diag_embed(drho_dPhi)

    dPhi = params.newton_raphson_rate * torch.linalg.solve(-J, F)
    dV = dPhi * -consts.Q_E
    V_el = q["V_el"] + dV

    # Correct V_el: Add a linear potential gradient to V_el s.t.
    # it matches the boundary potentials
    V_el_target_left = 0
    V_el_target_right = -q["voltage"] * consts.EV
    V_el_left = quantities.interpolate(
        V_el, q.grid, qs["boundary0"].grid, dimension_label="x"
    )
    V_el_right = quantities.interpolate(
        V_el,
        q.grid,
        qs[f"boundary{N}"].grid,
        dimension_label="x",
    )
    x_left = qs["boundary0"]["x"]
    x_right = qs[f"boundary{N}"]["x"]
    device_length = x_right - x_left
    left_factor = (x_right - q["x"]) / device_length
    right_factor = (q["x"] - x_left) / device_length
    corrected_V_el = (
        V_el
        + (V_el_target_left - V_el_left) * left_factor
        + (V_el_target_right - V_el_right) * right_factor
    )

    q["V_el_new"] = corrected_V_el

    return qs
