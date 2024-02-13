import torch

from kolpinn.mathematics import complex_abs2

import parameters as params
import physics


def get_SE_loss(q, *, with_grad, i):
    """
    i: layer index in [1,N]
    For a constant effective mass only if fd_second_derivatives!
    """

    # Multiplying hbar here for numerical stability
    hbar_phi_dx_over_m = q['phi_dx' + str(i)] * (physics.H_BAR / q['m_eff'+str(i)])

    if params.fd_second_derivatives:
        #hbar_phi_dx_over_m_dx = hbar_phi_dx_over_m.get_fd_derivative('x')
        phi_dx_dx = q[f'phi{i}'].get_fd_second_derivative('x')
        hbar_phi_dx_over_m_dx = phi_dx_dx * (physics.H_BAR / q['m_eff'+str(i)])
    else:
        hbar_phi_dx_over_m_dx = hbar_phi_dx_over_m.get_grad(
            q['x'],
            retain_graph=True,
            create_graph=with_grad,
        )
    residual = (-0.5 * physics.H_BAR * hbar_phi_dx_over_m_dx
                + (q['V'+str(i)] - q['E']) * q['phi'+str(i)])
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
        phi_dx_left = q['phi_dx'+left_index]
    if i==N: # Rightmost boundary
        a_r = q['phi'+left_index] - physics.B_R
        phi_dx_right = 1j * q['k'+right_index] * (a_r - physics.B_R)
    else:
        phi_dx_right = q['phi_dx'+right_index]
    residual = phi_dx_left / q['m_eff'+left_index] - phi_dx_right / q['m_eff'+right_index]
    residual /= physics.CURRENT_CONTINUITY_OOM

    return params.loss_function(residual)

def get_const_j_loss(q, *, with_grad, i):
    phi = q['phi'+str(i)]
    prob_current = torch.imag(physics.H_BAR * torch.conj(phi)
                              * q['phi_dx'+str(i)] / q['m_eff'+str(i)])
    residual = prob_current - prob_current.mean_dimension('x')
    residual /= physics.PROBABILITY_CURRENT_OOM
    # (j - j_mean) / j_mean
    #residual = prob_current / prob_current.mean_dimension('x') - 1

    return params.loss_function(residual)
