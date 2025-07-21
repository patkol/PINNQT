# Copyright (c) 2025 ETH Zurich, Patrice Kolb

# IDEA: turn these into single models (q_full is now passed through)


import torch

from kolpinn.mathematics import complex_abs2, grad
from kolpinn import quantities

import physical_constants as consts
import parameters as params


def SE_loss_trafo(qs, *, qs_full, with_grad, i, contact):
    """
    i: layer index in [1,N]
    For a constant effective mass only if fd_second_derivatives!
    """

    q = qs[f"bulk{i}"]
    q_full = qs_full[f"bulk{i}"]

    if params.fd_second_derivatives:
        if quantities.is_singleton_dimension("x", q[f"m_eff{i}"], q.grid):
            phi_dx_dx = quantities.get_fd_second_derivative(
                "x",
                q[f"phi{i}_{contact}"],
                q.grid,
            )
            phi_dx_over_m_dx = phi_dx_dx / q[f"m_eff{i}"]
        else:
            phi_dx_over_m_dx = quantities.get_fd_second_derivative(
                "x",
                q[f"phi{i}_{contact}"],
                q.grid,
                factor=1 / q[f"m_eff{i}"],
            )
    else:
        phi_dx_over_m_dx_full = grad(
            q[f"phi{i}_{contact}_dx"] / q[f"m_eff{i}"],
            q_full["x"],
            retain_graph=True,
            create_graph=with_grad,
        )
        phi_dx_over_m_dx = quantities.restrict(phi_dx_over_m_dx_full, q.grid)
    residual = (
        -0.5 * consts.H_BAR**2 * phi_dx_over_m_dx
        + (q[f"V_int{i}"] + q[f"V_el{i}"] - q[f"E_{contact}"]) * q[f"phi{i}_{contact}"]
    )
    # residual = -0.5 * consts.H_BAR * hbar_phi_dx_over_m_dx / q[f"phi{i}_{contact}"] + (
    #     q[f"V_int{i}"] + q[f"V_el{i}"] - q[f"E_{contact}"]
    # )
    # incoming_amplitude = complex_abs2(qs["bulk"][f"incoming_coeff_{contact}"])
    # residual /= incoming_amplitude
    residual /= qs["bulk"][f"incoming_coeff_{contact}"]
    residual /= params.V_OOM
    # Fermi-Dirac weighting
    # residual *= 1 / (
    #     1 + torch.exp(params.BETA * (q[f"E_{contact}"] - params.CONSTANT_FERMI_LEVEL))
    # )
    q[f"SE_loss{i}_{contact}"] = params.loss_function(residual)


def j_loss_trafo(qs, *, i, contact):
    """
    i = "" for averaging over the full device
    """

    q = qs[f"bulk{i}"]

    real_v_in = torch.real(qs[contact.grid_name][f"v{contact.index}_{contact}"])
    incoming_amplitude = complex_abs2(qs["bulk"][f"incoming_coeff_{contact}"])
    T = q[f"j{i}_{contact}"] / real_v_in / incoming_amplitude
    T_averaged = quantities.mean_dimension("x", T, q.grid)
    residual = T - T_averaged
    q[f"j_loss{i}_{contact}"] = params.loss_function(residual)


# IDEA: Fermi-Dirac weight the boundary losses
def wc_loss_trafo(qs, *, i, contact):
    # Satisfied at the contacts by definition
    q = qs[f"boundary{i}"]

    in_index = contact.get_in_layer_index(i)
    out_index = contact.get_out_layer_index(i)

    residual = q[f"phi{out_index}_{contact}"] - q[f"phi{in_index}_{contact}"]
    q[f"wc_loss{i}_{contact}"] = params.loss_function(residual)


def cc_loss_trafo(qs, *, i, contact):
    q = qs[f"boundary{i}"]

    in_index = contact.get_in_layer_index(i)
    out_index = contact.get_out_layer_index(i)

    if i == contact.in_boundary_index:
        a = qs["bulk"][f"incoming_coeff_{contact}"]
        r = qs["bulk"][f"reflected_coeff_{contact}"]
        phi_dx_in = contact.direction * 1j * q[f"k{in_index}_{contact}"] * (a - r)
    else:
        phi_dx_in = q[f"phi{in_index}_{contact}_dx"]
    if i == contact.out_boundary_index:
        t = qs["bulk"][f"transmitted_coeff_{contact}"]
        phi_dx_out = contact.direction * 1j * q[f"k{out_index}_{contact}"] * t
    else:
        phi_dx_out = q[f"phi{out_index}_{contact}_dx"]

    residual = phi_dx_in / q[f"m_eff{in_index}"] - phi_dx_out / q[f"m_eff{out_index}"]
    residual /= params.CURRENT_CONTINUITY_OOM
    q[f"cc_loss{i}_{contact}"] = params.loss_function(residual)
