# Copyright (c) 2024 ETH Zurich, Patrice Kolb

# OPTIM: don't return qs in trafos


from typing import Dict, Callable, Sequence, Optional
import numpy as np
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


def k_function(q: QuantityDict, i: int, contact: Contact) -> torch.Tensor:
    return physics.k_function(
        q[f"m_eff{i}"],
        q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el{i}"],
    )


def gaussian(x, sigma):
    return torch.exp(-(x**2) / (2 * sigma**2))


def smoother_function(x, smoothing_range):
    return x * (1 - gaussian(x, smoothing_range))


# smooth_k: Fixing the non-smoothness of k in V at E=V
def smooth_k_function(q: QuantityDict, i: int, contact: Contact) -> torch.Tensor:
    # IDEA: Do I need to remove V_el?
    return physics.k_function(
        q[f"m_eff{i}"],
        smoother_function(
            q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el{i}"],
            params.energy_smoothing_range,
        ),
    )


def transition_function(a, b, transition_exp):
    """
    Smoothly transition from a at x=0 to b at x->inf.
    transition_exp(x) = torch.exp(-(x-x_left) / transition_distance)
    """
    return transition_exp * a + (1 - transition_exp) * b


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
        raise ValueError("Unknown dx mode:", mode)

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


# For u phi0 + v conj(phi0)
# def get_phi_zero(q: QuantityDict, *, i: int, contact: Contact):
#     return (
#         q[f"a_output{i}_{contact}"] * q[f"a_phase{i}_{contact}"]
#         + q[f"b_output{i}_{contact}"] * q[f"b_phase{i}_{contact}"]
#     )


# For d0 phi0 + d1 phi1
def get_phi_zero(q: QuantityDict, *, i: int, contact: Contact):
    return q[f"a_output{i}_{contact}"] * q[f"a_phase{i}_{contact}"]


def get_phi_one(q: QuantityDict, *, i: int, contact: Contact):
    return q[f"b_output{i}_{contact}"] * q[f"b_phase{i}_{contact}"]


def get_phi_target(q: QuantityDict, *, i: int, contact: Contact):
    """
    Boundary conditions: Given phi{next_i}, find phi{i} at the boundary between
    i and next_i.
    """
    next_i = i + contact.direction
    phi_target = q[f"phi{next_i}_{contact}"]
    phi_dx_target = (
        q[f"m_eff{i}"] / q[f"m_eff{next_i}"] * q[f"phi{next_i}_{contact}_dx"]
    )

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


def get_V_electrostatic(qs, *, contacts) -> tuple[torch.Tensor, Grid]:
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

    return V_el, q.grid


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
    integral_at_x_0 = torch.tensor(0)
    layer_indices = range(N, 0, -1) if contact.direction == 1 else range(1, N + 1)
    for i in layer_indices:
        grid_names = [f"bulk{i}"]
        for i_boundary in (i - 1, i):
            for dx_string in dx_dict.keys():
                grid_names.append(f"boundary{i_boundary}" + dx_string)
        child_grids: dict[str, Grid] = dict(
            (grid_name, qs[grid_name].grid) for grid_name in grid_names
        )
        supergrid = Supergrid(child_grids, "x", copy_all=False)
        ks = [qs[grid_name][f"k{i}_{contact}"] for grid_name in grid_names]
        integrand = quantities.combine_quantity(
            ks, list(supergrid.subgrids.values()), supergrid
        )
        out_boundary_index = contact.get_out_boundary_index(i)
        x_0 = qs[f"boundary{out_boundary_index}"].grid["x"].item()
        integral = quantities.get_cumulative_integral(
            "x", x_0, integrand, supergrid, start_value=integral_at_x_0
        )

        for grid_name in grid_names:
            q = qs[grid_name]
            q[f"k_integral{i}_{contact}"] = quantities.restrict(
                integral, supergrid.subgrids[grid_name]
            )
            q[f"a_phase{i}_{contact}"] = torch.exp(1j * q[f"k_integral{i}_{contact}"])
            q[f"b_phase{i}_{contact}"] = torch.exp(-1j * q[f"k_integral{i}_{contact}"])

        # q_in_boundary = qs[f"boundary{contact.get_in_boundary_index(i)}"]
        # boundary_slice = grids.get_nd_slice("x", 0, q_in_boundary.grid)
        # integral_at_x_0 = q_in_boundary[f"k_integral{i}_{contact}"][boundary_slice]

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


# u phi0 + v conj(phi0)
# def phi_trafo(qs, *, i: int, contact: Contact, grid_names: Sequence[str]):
#     boundary_out = f"boundary{contact.get_out_boundary_index(i)}"
#     q_out = qs[boundary_out]
#     phi_zero = q_out[f"phi_zero{i}_{contact}"]
#     phi_zero_dx = q_out[f"phi_zero{i}_{contact}_dx"]
#     phi_target, phi_dx_target = get_phi_target(q_out, i=i, contact=contact)
#
#     # phi = u * phi_zero + v * conj(phi_zero), find u & v s.t. this matches the target
#     determinant = 2j * torch.imag(phi_zero * torch.conj(phi_zero_dx))
#     u = (
#         phi_target * torch.conj(phi_zero_dx) - torch.conj(phi_zero) * phi_dx_target
#     ) / determinant
#     v = -(phi_target * phi_zero_dx - phi_zero * phi_dx_target) / determinant
#
#     for grid_name in grid_names:
#         q = qs[grid_name]
#         phi_zero_full = q[f"phi_zero{i}_{contact}"]
#         q[f"phi{i}_{contact}"] = u * phi_zero_full + v * torch.conj(phi_zero_full)
#
#     return qs


# d0 phi0 + d1 phi1
def phi_trafo(qs, *, i: int, contact: Contact, grid_names: Sequence[str]):
    boundary_out = f"boundary{contact.get_out_boundary_index(i)}"
    q_out = qs[boundary_out]
    phi_zero = q_out[f"phi_zero{i}_{contact}"]
    phi_zero_dx = q_out[f"phi_zero{i}_{contact}_dx"]
    phi_one = q_out[f"phi_one{i}_{contact}"]
    phi_one_dx = q_out[f"phi_one{i}_{contact}_dx"]
    phi_target, phi_dx_target = get_phi_target(q_out, i=i, contact=contact)

    determinant = phi_zero * phi_one_dx - phi_zero_dx * phi_one
    d_zero = (phi_one_dx * phi_target - phi_one * phi_dx_target) / determinant
    d_one = (-phi_zero_dx * phi_target + phi_zero * phi_dx_target) / determinant

    for grid_name in grid_names:
        q = qs[grid_name]
        phi_zero_full = q[f"phi_zero{i}_{contact}"]
        phi_one_full = q[f"phi_one{i}_{contact}"]
        q[f"phi{i}_{contact}"] = d_zero * phi_zero_full + d_one * phi_one_full

    return qs


def contact_coeffs_trafo(qs, *, contact):
    q_in = qs[contact.grid_name]
    i = contact.index
    phi_target, phi_dx_target = get_phi_target(q_in, i=i, contact=contact)
    k = q_in[f"k{i}_{contact}"]
    q = qs["bulk"]
    q[f"incoming_coeff_{contact}"] = 0.5 * (
        phi_target + phi_dx_target / (1j * contact.direction * k)
    )
    q[f"reflected_coeff_{contact}"] = phi_target - q[f"incoming_coeff_{contact}"]

    return qs


def TR_trafo(qs, *, contact):
    """Transmission probability"""
    q = qs["bulk"]
    q_in = qs[contact.grid_name]
    q_out = qs[contact.out_boundary_name]
    incoming_amplitude = complex_abs2(q[f"incoming_coeff_{contact}"])
    reflected_amplitude = complex_abs2(q[f"reflected_coeff_{contact}"])
    transmitted_amplitude = complex_abs2(q[f"transmitted_coeff_{contact}"])
    real_v_in = torch.real(q_in[f"v{contact.index}_{contact}"])
    real_v_out = torch.real(q_out[f"v{contact.out_index}_{contact}"])
    q[f"T_{contact}"] = (
        transmitted_amplitude / incoming_amplitude * real_v_out / real_v_in
    )
    q[f"R_{contact}"] = reflected_amplitude / incoming_amplitude

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
    assert torch.allclose(
        torch.imag(v), torch.tensor(0, dtype=params.si_real_dtype)
    ), "The energy in the input contact should be positive"
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
