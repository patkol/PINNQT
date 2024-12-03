# Copyright (c) 2024 ETH Zurich, Patrice Kolb

# IDEA: turn these into single models (q_full is now passed through)

import torch

from kolpinn.mathematics import grad
from kolpinn.quantities import get_fd_second_derivative, mean_dimension, restrict

import physical_constants as consts
import parameters as params
import physics


def SE_loss_trafo(qs, *, qs_full, with_grad, i, N, contact):
    """
    i: layer index in [1,N]
    For a constant effective mass only if fd_second_derivatives!
    """

    q = qs[f"bulk{i}"]
    q_full = qs_full[f"bulk{i}"]

    if params.fd_second_derivatives:
        phi_dx_dx = get_fd_second_derivative("x", q[f"phi{i}_{contact}"], q.grid)
        hbar_phi_dx_over_m_dx = phi_dx_dx * (consts.H_BAR / q[f"m_eff{i}"])
    else:
        # Multiplying hbar here for numerical stability
        hbar_phi_dx_over_m = q[f"phi{i}_{contact}_dx"] * (consts.H_BAR / q[f"m_eff{i}"])
        hbar_phi_dx_over_m_dx_full = grad(
            hbar_phi_dx_over_m,
            q_full["x"],
            retain_graph=True,
            create_graph=with_grad,
        )
        hbar_phi_dx_over_m_dx = restrict(hbar_phi_dx_over_m_dx_full, q.grid)
    residual = (
        -0.5 * consts.H_BAR * hbar_phi_dx_over_m_dx
        + (q[f"V_int{i}"] + q[f"V_el{i}"] - q[f"E_{contact}"]) * q[f"phi{i}_{contact}"]
    )
    # residual = -0.5 * consts.H_BAR * hbar_phi_dx_over_m_dx / q[f"phi{i}_{contact}"] + (
    #     q[f"V_int{i}"] + q[f"V_el{i}"] - q[f"E_{contact}"]
    # )
    # incoming_amplitude = complex_abs2(qs["bulk"][f"incoming_coeff_{contact}"])
    # residual /= incoming_amplitude
    # residual /= qs["bulk"][f"incoming_coeff_{contact}"]
    residual /= params.V_OOM
    # Fermi-Dirac weighting
    residual *= 1 / (
        1 + torch.exp(physics.BETA * (q[f"E_{contact}"] - params.CONSTANT_FERMI_LEVEL))
    )
    q[f"SE_loss{i}_{contact}"] = params.loss_function(residual)

    return qs


def j_loss_trafo(qs, *, i, N, contact):
    q = qs[f"bulk{i}"]

    prob_current = q[f"j{i}_{contact}"]
    residual = prob_current - mean_dimension("x", prob_current, q.grid)
    # residual /= complex_abs2(qs["bulk"][f"incoming_coeff_{contact}"])
    # residual /= mean_dimension("x", complex_abs2(q[f"phi{i}_{contact}"]), q.grid)
    residual /= params.PROBABILITY_CURRENT_OOM
    # exact_prob_current = qs["bulk"][f"j_exact_{contact}"]
    # residual = torch.log(complex_abs2(prob_current / exact_prob_current))
    # Fermi-Dirac weighting
    residual *= 1 / (
        1 + torch.exp(physics.BETA * (q[f"E_{contact}"] - params.CONSTANT_FERMI_LEVEL))
    )
    q[f"j_loss{i}_{contact}"] = params.loss_function(residual)

    return qs


def wc_loss_trafo(qs, *, i, contact):
    # Satisfied at the contacts by definition
    q = qs[f"boundary{i}"]

    in_index = contact.get_in_layer_index(i)
    out_index = contact.get_out_layer_index(i)

    residual = q[f"phi{out_index}_{contact}"] - q[f"phi{in_index}_{contact}"]
    q[f"wc_loss{i}_{contact}"] = params.loss_function(residual)

    return qs


def cc_loss_trafo(qs, *, i, contact):
    q = qs[f"boundary{i}"]

    in_index = contact.get_in_layer_index(i)
    out_index = contact.get_out_layer_index(i)

    if i == contact.in_boundary_index:
        r = qs["bulk"][f"reflected_coeff_{contact}"]
        phi_dx_in = 1j * q[f"k{in_index}_{contact}"] * (1 - r)
    else:
        phi_dx_in = q[f"phi{in_index}_{contact}_dx"]
    if i == contact.out_boundary_index:
        t = qs["bulk"][f"transmitted_coeff_{contact}"]
        phi_dx_out = 1j * q[f"k{out_index}_{contact}"] * t
    else:
        phi_dx_out = q[f"phi{out_index}_{contact}_dx"]

    residual = phi_dx_in / q[f"m_eff{in_index}"] - phi_dx_out / q[f"m_eff{out_index}"]
    residual /= params.CURRENT_CONTINUITY_OOM
    residual *= 1e1
    q[f"cc_loss{i}_{contact}"] = params.loss_function(residual)

    return qs
