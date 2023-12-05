import torch

from kolpinn.mathematics import complex_abs2

import parameters as params
import physics


def get_SE_loss(q, *, with_grad, i):
    """
    i: layer index in [1,N]
    """
    phi_dx = q['phi'+str(i)].get_grad(
        q['x'],
        retain_graph=True,
        create_graph=True,
    )
    # Multiplying hbar here for numerical stability
    hbar_phi_dx_over_m = phi_dx * (physics.H_BAR / q['m_eff'+str(i)])
    hbar2_phi_dx_over_m_dx = hbar_phi_dx_over_m.get_grad(
        q['x'],
        retain_graph=True,
        create_graph=with_grad,
    ) * physics.H_BAR
    residual = -0.5 * hbar2_phi_dx_over_m_dx + (q['V'+str(i)] - q['E']) * q['phi'+str(i)]
    residual /= physics.V_OOM

    return params.loss_function(residual)

def get_wc_loss(q, *, with_grad, i):
    left_index = str(i)
    right_index = str(i+1)
    residual = q['phi' + left_index] - q['phi' + right_index]

    return params.loss_function(residual)

def get_cc_loss(q, *, with_grad, i, N):
    left_index = str(i)
    right_index = str(i+1)
    if i==0: # Leftmost boundary
        # OPTIM: directly calculate phi_dx_left/right at boundaries
        b_l = q['phi'+right_index] - physics.A_L
        phi_dx_left = 1j * q['k'+left_index] * (physics.A_L - b_l)
    else:
        phi_dx_left = q['phi'+left_index].get_grad(
            q['x'],
            retain_graph=True,
            create_graph=with_grad,
        )
    if i==N: # Rightmost boundary
        b_r = q['phi'+left_index] - physics.A_R
        phi_dx_right = 1j * q['k'+right_index] * (b_r - physics.A_R)
    else:
        phi_dx_right = q['phi'+right_index].get_grad(
            q['x'],
            retain_graph=True,
            create_graph=with_grad,
        )
    residual = phi_dx_left / q['m_eff'+left_index] - phi_dx_right / q['m_eff'+right_index]
    residual /= physics.CURRENT_CONTINUITY_OOM

    return params.loss_function(residual)

def get_const_j_loss(q, *, with_grad, i):
    phi = q['phi'+str(i)]
    # OPTIM: same phi_dx in SE loss
    phi_dx = phi.get_grad(
        q['x'],
        retain_graph=True,
        create_graph=with_grad,
    )
    prob_current_complex = physics.H_BAR * phi.transform(torch.conj) * phi_dx / q['m_eff'+str(i)]
    prob_current = prob_current_complex.transform(torch.imag)
    residual = prob_current - prob_current.mean_dimension('x')
    residual /= physics.PROBABILITY_CURRENT_OOM
    # (j - j_mean) / j_mean
    #residual = prob_current / prob_current.mean_dimension('x') - 1

    return params.loss_function(residual)
