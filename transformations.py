# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Dict, Callable
import numpy as np
import torch

from kolpinn import mathematics
from kolpinn.mathematics import complex_abs2
from kolpinn.grids import Subgrid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import model

import physics
from classes import Contact


def k_function(q: QuantityDict, i: int, contact: Contact) -> torch.Tensor:
    return physics.k_function(
        q[f"m_eff{i}"],
        q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el_approx{i}"],
    )


def gaussian(x, sigma):
    return torch.exp(-(x**2) / (2 * sigma**2))


def smoother_function(x, smoothing_range):
    return x * (1 - gaussian(x, smoothing_range))


# smooth_k: Fixing the non-smoothness of k in V at E=V
def smooth_k_function(q: QuantityDict, i: int, contact: Contact) -> torch.Tensor:
    return physics.k_function(
        q[f"m_eff{i}"],
        smoother_function(
            q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el_approx{i}"],
            physics.energy_smoothing_range,
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
    V_voltage = -q["voltage"] * physics.EV * distance_factor

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
            qs[grid_name][name] = (quantity_right - quantity_left) / (2 * physics.dx)
            return qs

    elif mode == "singlegrid":

        def dx_qs_trafo(qs):
            q = qs[grid_name]
            q[name] = quantities.get_fd_derivative("x", q[quantity_name], q.grid)
            return qs

    else:
        raise ValueError("Unknown dx mode:", mode)

    return model.MultiModel(dx_qs_trafo, name)


def get_E_fermi(q: QuantityDict, *, i):
    dos = 8 * np.pi / physics.H**3 * torch.sqrt(2 * q[f"m_eff{i}"] ** 3 * q["DeltaE"])

    best_E_f = None
    best_abs_remaining_charge = float("inf")
    E_fs = np.arange(0 * physics.EV, 0.8 * physics.EV, 1e-3 * physics.EV)
    for E_f in E_fs:
        fermi_dirac = 1 / (1 + torch.exp(physics.BETA * (q["DeltaE"] - E_f)))
        integrand = dos * fermi_dirac
        # Particle, not charge density
        n = quantities.sum_dimension("DeltaE", integrand, q.grid) * physics.E_STEP
        abs_remaining_charge = torch.abs(n - q[f"doping{i}"]).item()
        if abs_remaining_charge < best_abs_remaining_charge:
            best_abs_remaining_charge = abs_remaining_charge
            best_E_f = E_f

        E_f += 1e-3 * physics.EV

    assert best_E_f is not None
    assert best_E_f != E_fs[0] and best_E_f != E_fs[-1], E_f / physics.EV

    relative_remaining_charge = best_abs_remaining_charge / q[f"doping{i}"]
    print(
        f"Best fermi energy: {best_E_f / physics.EV} eV with a relative remaining charge of {relative_remaining_charge}"
    )

    # best_E_f = 0.258 * physics.EV # Fixed fermi level

    return best_E_f + q[f"V_int{i}"] + q[f"V_el{i}"]


def factors_trafo(qs, i, contact):
    """
    Use the boundary conditions to find the factors to multiply a/b_output with
    given that a and b are known in layer `i_next`
    """

    i_next = contact.get_next_layer_index(i)
    i_boundary = contact.get_out_boundary_index(i)
    q = qs[f"boundary{i_boundary}"]

    m = q[f"m_eff{i}"]
    m_next = q[f"m_eff{i_next}"]
    k = q[f"smooth_k{i}_{contact}"]
    k_next = q[f"smooth_k{i_next}_{contact}"]
    o_a = q[f"a_output{i}_{contact}"]
    o_b = q[f"b_output{i}_{contact}"]
    z_a = 1j * k * o_a + q[f"a_output{i}_{contact}_dx"]
    z_b = 1j * k * o_b - q[f"b_output{i}_{contact}_dx"]
    a_next = q[f"a{i_next}_propagated_{contact}"]
    b_next = q[f"b{i_next}_propagated_{contact}"]
    a_dx_next = q[f"a{i_next}_propagated_{contact}_dx"]
    b_dx_next = q[f"b{i_next}_propagated_{contact}_dx"]

    a_factor = (
        a_next
        + b_next
        + m
        / m_next
        * o_b
        / z_b
        * (a_dx_next + b_dx_next + 1j * k_next * (a_next - b_next))
    ) / (o_a + z_a / z_b * o_b)

    b_factor = (a_next + b_next - a_factor * o_a) / o_b

    q[f"a_factor{i}_{contact}"] = a_factor
    q[f"b_factor{i}_{contact}"] = b_factor

    return qs


def add_coeff(
    c: str,
    qs: dict[str, QuantityDict],
    contact: Contact,
    grid_name: str,
    i: int,
):
    boundary_q = qs[f"boundary{contact.get_out_boundary_index(i)}"]
    q = qs[grid_name]

    coeff = boundary_q[f"{c}_factor{i}_{contact}"] * q[f"{c}_output{i}_{contact}"]
    q[f"{c}{i}_{contact}"] = coeff

    return qs


def add_coeffs(
    qs: dict[str, QuantityDict],
    contact: Contact,
    grid_name: str,
    i: int,
):
    for c in ("a", "b"):
        add_coeff(c, qs, contact, grid_name, i)

    return qs


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
    Inverse of 'to_full_trafo'
    """
    full_quantity = qs["bulk"][quantity_label]
    for i in range(1, N + 1):
        q_layer = qs[f"bulk{i}"]
        assert isinstance(q_layer.grid, Subgrid)
        q_layer[label_fn(i)] = quantities.restrict(full_quantity, q_layer.grid)

    return qs


def dos_trafo(qs, *, contact):
    q = qs["bulk"]
    q_in = qs[contact.grid_name]

    incoming_coeff_in = q_in[
        f"{contact.incoming_coeff_in_name}{contact.index}_{contact}"
    ]
    incoming_amplitude = complex_abs2(incoming_coeff_in)
    q[f"DOS_{contact}"] = (
        1
        / (2 * np.pi)
        * complex_abs2(q[f"phi_{contact}"])
        / q_in[f"dE_dk{contact.index}_{contact}"]
        / incoming_amplitude
    )
    return qs


def n_contact_trafo(qs, *, contact):
    """Density from one contact only"""
    q = qs["bulk"]
    q_in = qs[contact.grid_name]

    integrand = q[f"DOS_{contact}"] * q_in[f"fermi_integral_{contact}"]
    q[f"n_{contact}"] = (
        quantities.sum_dimension("DeltaE", integrand, q.grid) * physics.E_STEP
    )
    return qs


def n_trafo(qs, *, contacts):
    """Full density"""
    q = qs["bulk"]
    q["n"] = sum(q[f"n_{contact}"] for contact in contacts)

    return qs


def V_electrostatic_trafo(qs):
    q = qs["bulk"]

    assert q.grid.dimensions_labels[-1] == "x"

    # Construct the discretized Laplace operator M assuming an
    # equispaced x grid
    # OPTIM: do this only once
    Nx = q.grid.dim_size["x"]
    dx = q.grid.dimensions["x"][1] - q.grid.dimensions["x"][0]
    M = torch.zeros(Nx, Nx)
    permittivity = quantities.squeeze_to(["x"], q["permittivity"], q.grid)
    for i in range(1, Nx):
        M[i, i - 1] = (permittivity[i] + permittivity[i - 1]) / (2 * dx**2)
    for i in range(0, Nx - 1):
        M[i, i + 1] = (permittivity[i] + permittivity[i + 1]) / (2 * dx**2)
    for i in range(1, Nx - 1):
        M[i, i] = -M[i, i - 1] - M[i, i + 1]

    # Dirichlet BC
    # Assuming constant permittivity at the outer boundaries
    M[0, 0] = -2 * permittivity[0] / dx**2
    M[-1, -1] = -2 * permittivity[-1] / dx**2

    # TODO: implementation that works if x is not the last coordinate,
    #       kolpinn function
    #       Then remove the assertion above
    rho = physics.Q_E * (q["doping"] - q["n"])
    rhs = -physics.Q_E * rho
    # Phi[-1] = 0, Phi[Nx] = -voltage * EV
    # S[0] = -epsilon[-1/2] / dx^2 * Phi[-1],
    # S[Nx-1] = -epsilon[Nx-1/2] / dx^2 * Phi[Nx]
    Phi_Nx = q["voltage"].squeeze(-1) * physics.EV
    rhs[:, :, Nx - 1] += -permittivity[-1] / dx**2 * Phi_Nx
    # Permute: x must be the second last coordinate for torch.linalg.solve
    rhs_permuted = torch.permute(rhs, (0, 2, 1))
    Phi_permuted = torch.linalg.solve(M, rhs_permuted)
    Phi = torch.permute(Phi_permuted, (0, 2, 1))
    q["V_el"] = -physics.Q_E * Phi

    return qs


def TR_trafo(qs, *, contact):
    """Transmission probability"""
    q = qs["bulk"]
    q_in = qs[contact.grid_name]
    q_out = qs[contact.out_boundary_name]
    incoming_coeff_in = q_in[
        f"{contact.incoming_coeff_in_name}{contact.index}_{contact}"
    ]
    incoming_coeff_out = q_in[
        f"{contact.incoming_coeff_out_name}{contact.index}_{contact}"
    ]
    outgoing_coeff_out = q_out[
        f"{contact.outgoing_coeff_out_name}{contact.out_index}_propagated_{contact}"
    ]
    real_v_in = torch.real(q_in[f"v{contact.index}_{contact}"])
    real_v_out = torch.real(q_out[f"v{contact.out_index}_{contact}"])
    q[f"T_{contact}"] = (
        complex_abs2(outgoing_coeff_out)
        / complex_abs2(incoming_coeff_in)
        * real_v_out
        / real_v_in
    )
    q[f"R_{contact}"] = complex_abs2(incoming_coeff_out) / complex_abs2(
        incoming_coeff_in
    )

    return qs


def I_contact_trafo(qs, *, contact):
    q = qs["bulk"]
    q_in = qs[contact.grid_name]
    integrand = q[f"T_{contact}"] * q_in[f"fermi_integral_{contact}"]
    integral = quantities.sum_dimension("DeltaE", integrand, q.grid) * physics.E_STEP
    sign = 1 if contact.name == "L" else -1
    prefactor = -physics.Q_E / physics.H_BAR / (2 * np.pi) * sign
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
    sign = -1 if contact.name == "L" else 1
    q[f"j_exact_{contact}"] = sign * physics.H_BAR * k / m_eff

    return qs
