# Copyright (c) 2025 ETH Zurich, Patrice Kolb

# OPTIM: don't return qs in trafos


from typing import Dict, Callable, Sequence, Optional
import numpy as np
import torch

from kolpinn.mathematics import complex_abs2
from kolpinn import grids
from kolpinn.grids import Grid, Subgrid, Supergrid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict

from classes import Contact
import physical_constants as consts
import parameters as params
import formulas


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
        sorted_k_integral = formulas.smoothen(sorted_k_integral, sorted_supergrid, "x")
        k_integral_LR = quantities.combine_quantity(
            [sorted_k_integral], [sorted_supergrid], supergrid
        )
        right_k_integral = quantities.restrict(
            k_integral_LR, supergrid.subgrids[f"boundary{right_boundary_index}"]
        )
        # k_integral_RL: integrating from x_R (k_integral integrates from x_L)
        k_integral_RL = k_integral_LR - right_k_integral

        if contact.direction == 1:
            a_phase = torch.exp(1j * k_integral_LR)
            # Conjugate k for the calculation of b_phase to ensure
            # that it decays towards the output
            b_phase = torch.exp(-1j * torch.conj(k_integral_LR))

        else:
            a_phase = torch.exp(-1j * k_integral_RL)
            b_phase = torch.exp(1j * torch.conj(k_integral_RL))

        if params.ansatz == "half_wkb":
            b_phase = torch.ones_like(b_phase)

        if params.ignore_wkb_phase:
            a_phase = torch.abs(a_phase)
            b_phase = torch.abs(b_phase)

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
        formulas.get_E_fermi(q_in, i=i) + q_in[f"V_int{i}"] + q_in[f"V_el{i}"]
    )

    return qs


def fermi_integral_trafo(qs, *, contact: Contact):
    q = qs["bulk"]
    q_in = qs[contact.grid_name]
    i = contact.index
    qs["bulk"][f"fermi_integral_{contact}"] = formulas.get_fermi_integral(
        m_eff=q_in[f"m_eff{i}"], E_fermi=q[f"E_fermi_{contact}"], E=q[f"E_{contact}"]
    )

    return qs


def phi_zero_one_trafo(qs, *, i: int, contact: Contact, grid_names: Sequence[str]):
    """
    Add phi_zero and phi_one.
    """

    # x_in = qs[f"boundary{contact.in_boundary_index}"]["x"]
    x_in = qs[f"boundary{contact.get_in_boundary_index(i)}"]["x"]

    for grid_name in grid_names:
        q = qs[grid_name]
        a_output = q[f"a_output{i}_{contact}"]
        a_output = formulas.a_output_transformation(
            a_output, q=q, x_in=x_in, i=i, contact=contact
        )
        q[f"phi_zero{i}_{contact}"] = a_output * q[f"a_phase{i}_{contact}"]

        if not params.use_phi_one:
            continue

        b_output = q[f"b_output{i}_{contact}"]
        b_output = formulas.b_output_transformation(
            b_output, q=q, x_in=x_in, i=i, contact=contact
        )
        q[f"phi_one{i}_{contact}"] = b_output * q[f"b_phase{i}_{contact}"]

    return qs


def simple_phi_trafo(qs, *, i: int, contact: Contact, grid_names: Sequence[str]):
    """phi0 + phi1 (BC not forced)"""

    for grid_name in grid_names:
        q = qs[grid_name]
        q[f"phi{i}_{contact}"] = formulas.get_simple_phi(q, i=i, contact=contact)

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
    phi_target, phi_dx_target = formulas.get_phi_target(
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
    phi_target, phi_dx_target = formulas.get_phi_target(
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
    Condition: phi' = gamma * (2a - phi) at the input contact.
    with gamma := contact.direction * m_eff_device / m_eff_contact * i * k_contact.
    Solved by d * phi_old
    with d = 2a / (phi_old_contact' / gamma + phi_old_contact)
    """

    assert direction == 1  # Don't need this special trafo for output -> input
    assert i == contact.in_layer_index

    boundary_index = contact.in_boundary_index
    q_boundary = qs[f"boundary{boundary_index}"]

    phi_boundary = formulas.get_simple_phi(q_boundary, i=i, contact=contact)
    phi_dx_boundary = formulas.get_simple_phi_dx(q_boundary, i=i, contact=contact)
    gamma = (
        contact.direction
        * q_boundary[f"m_eff{i}"]
        / q_boundary[f"m_eff{contact.index}"]
        * 1j
        * q_boundary[f"k{contact.index}_{contact}"]
    )
    a = qs["bulk"][f"incoming_coeff_{contact}"]
    d = 2 * a / (phi_dx_boundary / gamma + phi_boundary)

    for grid_name in grid_names:
        q = qs[grid_name]
        phi = formulas.get_simple_phi(q, i=i, contact=contact)
        q[f"phi{i}_{contact}"] = d * phi

    return qs


def hard_bc_out_phi_trafo(
    qs, *, i: int, contact: Contact, grid_names: Sequence[str], direction: int
):
    """
    Force the output BC at the output boundary without affecting the WF
    at the input boundary.
    BC on the output boundary: There can only be an
    outgoing wave there. The criterion for that is contact.direction*i*k*phi = phi_dx.
    Assuming that there the eff mass & potential plateau towards the
    contacts - then the solution should approach t * e^{contact.direction*ikx}.
    As t we use the value of the current phi at the output.
    """

    assert direction == 1  # Don't need this special trafo for output -> input
    assert i == contact.out_layer_index

    q_in_boundary = qs[f"boundary{contact.get_in_boundary_index(i)}"]
    q_out_boundary = qs[f"boundary{contact.get_out_boundary_index(i)}"]
    x_in, x_out = q_in_boundary["x"].item(), q_out_boundary["x"].item()

    max_transition_distance = abs(x_out - x_in)
    assert (
        params.hard_bc_output_transition_distance <= max_transition_distance
    ), f'The "hard_bc_output_transition_distance" is too wide, must be less than {max_transition_distance / consts.NM} nm for the current device'

    for grid_name in grid_names:
        q = qs[grid_name]
        delta_x = q["x"] - q_out_boundary["x"]
        plane_wave_phase = torch.exp(
            contact.direction * 1j * q_out_boundary[f"k{i}_{contact}"] * delta_x
        )
        # TODO: Force the correct transmitted amplitude based on the reflected one
        phi_out = q_out_boundary[f"phi{i}_{contact}"] * plane_wave_phase

        # Transition to phi_out
        transition_start = (
            x_out - contact.direction * params.hard_bc_output_transition_distance
        )
        transition_function = formulas.smooth_transition(
            q["x"], transition_start, x_out, 1, 0
        )
        q.overwrite(
            f"phi{i}_{contact}",
            transition_function * q[f"phi{i}_{contact}"]
            + (1 - transition_function) * phi_out,
        )

    return qs


def phi_trafo_learn_phi_prime(
    qs, i: int, contact: Contact, grid_names: Sequence[str], direction: int
):
    """Interpret phi_zero/one as 1/m phi'"""

    # Need some values to start the integration, naturally leading to hard bc
    assert direction in (1, -1)

    for grid_name in grid_names:
        q = qs[grid_name]
        phi_dx_over_m = q[f"phi_zero{i}_{contact}"]
        if params.use_phi_one:
            phi_dx_over_m += q[f"phi_one{i}_{contact}"]
        q[f"phi{i}_{contact}_dx"] = q[f"m_eff{i}"] * phi_dx_over_m

    boundary_index = (
        contact.get_in_boundary_index(i)
        if direction == 1
        else contact.get_out_boundary_index(i)
    )
    q_boundary = qs[f"boundary{boundary_index}"]
    x_boundary = q_boundary.grid["x"].item()

    if direction == 1 and i == contact.in_layer_index:
        gamma = (
            contact.direction
            * q_boundary[f"m_eff{i}"]
            / q_boundary[f"m_eff{contact.index}"]
            * 1j
            * q_boundary[f"k{contact.index}_{contact}"]
        )
        a = qs["bulk"][f"incoming_coeff_{contact}"]
        phi_dx_boundary = q_boundary[f"phi{i}_{contact}_dx"]
        phi_boundary = 2 * a - phi_dx_boundary / gamma

    else:
        prev_layer_index = (
            contact.get_in_layer_index(boundary_index)
            if direction == 1
            else contact.get_out_layer_index(boundary_index)
        )
        # Find the phi to start the integration from
        phi_boundary = q_boundary[f"phi{prev_layer_index}_{contact}"]
        # Find a correction factor d to phi_dx s.t. cc is satisfied
        phi_dx_boundary_current = q_boundary[f"phi{i}_{contact}_dx"]
        phi_dx_boundary_target = (
            q_boundary[f"phi{prev_layer_index}_{contact}_dx"]
            * q_boundary[f"m_eff{i}"]
            / q_boundary[f"m_eff{prev_layer_index}"]
        )
        d = phi_dx_boundary_target / phi_dx_boundary_current
        for grid_name in grid_names:
            q = qs[grid_name]
            q.overwrite(f"phi{i}_{contact}_dx", d * q[f"phi{i}_{contact}_dx"])

    # Integrate phi'
    # Code duplication w/ WKB ansatz
    child_grids: dict[str, Grid] = dict(
        (grid_name, qs[grid_name].grid) for grid_name in grid_names
    )
    supergrid = Supergrid(child_grids, "x", copy_all=False)
    sorted_supergrid = grids.get_sorted_grid_along(["x"], supergrid, copy_all=False)
    phi_dxs = [qs[grid_name][f"phi{i}_{contact}_dx"] for grid_name in grid_names]
    integrand = quantities.combine_quantity(
        phi_dxs, list(supergrid.subgrids.values()), supergrid
    )
    sorted_integrand = quantities.restrict(integrand, sorted_supergrid)
    sorted_integral = quantities.get_cumulative_integral(
        "x", x_boundary, sorted_integrand, sorted_supergrid
    )
    integral = quantities.combine_quantity(
        [sorted_integral], [sorted_supergrid], supergrid
    )
    integral += phi_boundary

    for grid_name in grid_names:
        q = qs[grid_name]
        q[f"phi{i}_{contact}"] = quantities.restrict(
            integral, supergrid.subgrids[grid_name]
        )

    return qs


def phi_trafo_learn_phi_prime_polar(
    qs, i: int, contact: Contact, grid_names: Sequence[str], direction: int
):
    """
    Interpret abs/angle[phi_zero] as abs(phi)' / angle(phi)'
    No hard BC at the input/output boundaries for direction == 1!
    Also there's no hard cc BC.
    phi_dx is not determined in here.
    """

    # Need some values to start the integration, naturally leading to hard bc
    assert direction == -1  # in (1, -1)

    # Find the phi to start the integration from
    boundary_index = (
        contact.get_in_boundary_index(i)
        if direction == 1
        else contact.get_out_boundary_index(i)
    )
    q_boundary = qs[f"boundary{boundary_index}"]
    x_boundary = q_boundary.grid["x"].item()
    prev_layer_index = (
        contact.get_in_layer_index(boundary_index)
        if direction == 1
        else contact.get_out_layer_index(boundary_index)
    )
    phi_boundary = q_boundary[f"phi{prev_layer_index}_{contact}"]

    # Integrate phi'
    # Code duplication w/ WKB ansatz
    child_grids: dict[str, Grid] = dict(
        (grid_name, qs[grid_name].grid) for grid_name in grid_names
    )
    supergrid = Supergrid(child_grids, "x", copy_all=False)
    sorted_supergrid = grids.get_sorted_grid_along(["x"], supergrid, copy_all=False)

    input_labels = ["zero"]
    if params.use_phi_one:
        input_labels.append("one")
    integral = 0
    for input_label in input_labels:
        phi_abs_dxs = [
            torch.abs(qs[grid_name][f"phi_{input_label}{i}_{contact}"])
            for grid_name in grid_names
        ]
        phi_angle_dxs = [
            torch.angle(qs[grid_name][f"phi_{input_label}{i}_{contact}"])
            for grid_name in grid_names
        ]
        abs_integrand = quantities.combine_quantity(
            phi_abs_dxs, list(supergrid.subgrids.values()), supergrid
        )
        angle_integrand = quantities.combine_quantity(
            phi_angle_dxs, list(supergrid.subgrids.values()), supergrid
        )
        sorted_abs_integrand = quantities.restrict(abs_integrand, sorted_supergrid)
        sorted_angle_integrand = quantities.restrict(angle_integrand, sorted_supergrid)
        sorted_abs_integral = quantities.get_cumulative_integral(
            "x", x_boundary, sorted_abs_integrand, sorted_supergrid
        )
        sorted_angle_integral = quantities.get_cumulative_integral(
            "x", x_boundary, sorted_angle_integrand, sorted_supergrid
        )
        sorted_integral = sorted_abs_integral * torch.exp(1j * sorted_angle_integral)
        integral += quantities.combine_quantity(
            [sorted_integral], [sorted_supergrid], supergrid
        )

    integral += phi_boundary

    assert isinstance(integral, torch.Tensor)
    for grid_name in grid_names:
        q = qs[grid_name]
        q[f"phi{i}_{contact}"] = quantities.restrict(
            integral, supergrid.subgrids[grid_name]
        )

    return qs


def phi_trafo(qs, *, learn_phi_prime, **kwargs):
    if learn_phi_prime:
        return phi_trafo_learn_phi_prime(qs, **kwargs, direction=params.hard_bc_dir)

    if params.learn_phi_prime_polar:
        return phi_trafo_learn_phi_prime_polar(
            qs, **kwargs, direction=params.hard_bc_dir
        )

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
    phi_in, phi_dx_in = formulas.get_phi_target(
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
        phi_out, _ = formulas.get_phi_target(
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
    q[f"n_{contact}"] = formulas.get_n_contact(
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

    # # Dirichlet BC
    # M[0, 0] = -(3 * permittivity[0] + permittivity[1]) / (2 * dx**2)
    # M[-1, -1] = -(3 * permittivity[-1] + permittivity[-2]) / (2 * dx**2)

    # TODO: implementation that works if x is not the last coordinate,
    #       kolpinn function
    #       Then remove the assertion above
    rho = consts.Q_E * (q["doping"] - q["n"])

    # Dirichlet BC
    V_voltage = -q["voltage"] * consts.EV
    rho[..., -1] += (permittivity[-1] * V_voltage[..., 0]) / dx**2

    Phi = q["V_el"] / -consts.Q_E
    F = torch.einsum("ij,...j->...i", M, Phi) + rho

    # Get the density if the potential was shifted by dV
    n_pdVs = []
    for contact in contacts:
        q_in = qs[contact.grid_name]
        i = contact.index
        fermi_integral_pdV = formulas.get_fermi_integral(
            m_eff=q_in[f"m_eff{i}"],
            E_fermi=q[f"E_fermi_{contact}"],
            E=q[f"E_{contact}"] + params.dV_poisson,
        )
        n_pdV = formulas.get_n_contact(
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

    # q["V_el_new"] = V_el

    return qs
