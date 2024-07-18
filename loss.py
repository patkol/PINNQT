# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from kolpinn.mathematics import complex_abs2, grad
from kolpinn.quantities import get_fd_second_derivative, mean_dimension, restrict

import parameters as params
import physics


def SE_loss_trafo(qs, *, qs_full, with_grad, i, N, contact):
    """
    i: layer index in [1,N]
    For a constant effective mass only if fd_second_derivatives!
    """

    q = qs[f'bulk{i}']
    q_full = qs_full[f'bulk{i}']

    coeff_in = qs[contact.grid_name][f'{contact.incoming_coeff_in_name}{contact.index}_{contact}']

    if params.fd_second_derivatives:
        phi_dx_dx = get_fd_second_derivative('x', q[f'phi{i}_{contact}'], q.grid)
        hbar_phi_dx_over_m_dx = phi_dx_dx * (physics.H_BAR / q[f'm_eff{i}'])
    else:
        # Multiplying hbar here for numerical stability
        hbar_phi_dx_over_m = q[f'phi{i}_{contact}_dx'] * (physics.H_BAR / q[f'm_eff{i}'])
        hbar_phi_dx_over_m_dx_full = grad(
            hbar_phi_dx_over_m,
            q_full['x'],
            retain_graph=True,
            create_graph=with_grad,
        )
        hbar_phi_dx_over_m_dx = restrict(hbar_phi_dx_over_m_dx_full, q.grid)
    residual = (-0.5 * physics.H_BAR * hbar_phi_dx_over_m_dx
                + (q[f'V_int{i}'] + q[f'V_el{i}'] - q[f'E_{contact}'])
                # + (q[f'V_int{i}'] + q[f'V_el_approx{i}'] - q[f'E_{contact}'])
                  * q[f'phi{i}_{contact}'])
    residual /= coeff_in
    residual /= physics.V_OOM
    q[f'SE_loss{i}_{contact}'] = params.loss_function(residual)

    return qs


def j_loss_trafo(qs, *, i, N, contact):
    q = qs[f'bulk{i}']

    prob_current = q[f'j{i}_{contact}']
    residual = prob_current - mean_dimension('x', prob_current, q.grid)
    # coeff_in = qs[contact.grid_name][f'{contact.incoming_coeff_in_name}{contact.index}_{contact}']
    # residual /= complex_abs2(coeff_in)
    residual /= mean_dimension('x', complex_abs2(q[f'phi{i}_{contact}']), q.grid)
    residual /= physics.PROBABILITY_CURRENT_OOM
    # exact_prob_current = qs[contact.out_boundary_name][f'j_exact_{contact}']
    # residual = torch.log(complex_abs2(prob_current / exact_prob_current))
    q[f'j_loss{i}_{contact}'] = params.loss_function(residual)

    return qs


def wc_loss_trafo(qs, *, i, contact):
    q = qs[f'boundary{i}']

    left_index = str(i)
    right_index = str(i + 1)
    residual = q[f'phi{right_index}_{contact}'] - q[f'phi{left_index}_{contact}']
    q[f'wc_loss{i}_{contact}'] = params.loss_function(residual)

    return qs


def cc_loss_trafo(qs, *, i, contact):
    q = qs[f'boundary{i}']

    left_index = str(i)
    right_index = str(i + 1)
    residual = (q[f'phi{right_index}_{contact}_dx'] / q['m_eff' + left_index]
                - q[f'phi{left_index}_{contact}_dx'] / q['m_eff' + right_index])
    residual /= physics.CURRENT_CONTINUITY_OOM
    q[f'cc_loss{i}_{contact}'] = params.loss_function(residual)

    return qs
