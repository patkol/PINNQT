import torch

from kolpinn.mathematics import grad
from kolpinn.grid_quantities import get_fd_second_derivative, mean_dimension, restrict

import parameters as params
import physics


def get_SE_loss(q, *, q_full, with_grad, i):
    """
    i: layer index in [1,N]
    For a constant effective mass only if fd_second_derivatives!
    """

    # Multiplying hbar here for numerical stability
    hbar_phi_dx_over_m = q[f'phi{i}_dx'] * (physics.H_BAR / q[f'm_eff{i}'])

    if params.fd_second_derivatives:
        #hbar_phi_dx_over_m_dx = hbar_phi_dx_over_m.get_fd_derivative('x')
        phi_dx_dx = get_fd_second_derivative('x', q[f'phi{i}'], q.grid)
        hbar_phi_dx_over_m_dx = phi_dx_dx * (physics.H_BAR / q[f'm_eff{i}'])
    else:
        hbar_phi_dx_over_m_dx_full = grad(
            hbar_phi_dx_over_m,
            q_full['x'],
            retain_graph=True,
            create_graph=with_grad,
        )
        hbar_phi_dx_over_m_dx = restrict(hbar_phi_dx_over_m_dx_full, q.grid)
    residual = (-0.5 * physics.H_BAR * hbar_phi_dx_over_m_dx
                + (q[f'V{i}'] - q['E']) * q[f'phi{i}'])
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
        phi_dx_left = q['phi'+left_index+'_dx']
    if i==N: # Rightmost boundary
        a_r = q['phi'+left_index] - physics.B_R
        phi_dx_right = 1j * q['k'+right_index] * (a_r - physics.B_R)
    else:
        phi_dx_right = q['phi'+right_index+'_dx']
    residual = phi_dx_left / q['m_eff'+left_index] - phi_dx_right / q['m_eff'+right_index]
    residual /= physics.CURRENT_CONTINUITY_OOM

    return params.loss_function(residual)

def get_const_j_loss(q, *, with_grad, i):
    phi = q['phi'+str(i)]
    prob_current = torch.imag(physics.H_BAR * torch.conj(phi)
                              * q[f'phi{i}_dx'] / q[f'm_eff{i}'])
    residual = prob_current - mean_dimension('x', prob_current, q.grid)
    residual /= physics.PROBABILITY_CURRENT_OOM
    # (j - j_mean) / j_mean
    #residual = prob_current / prob_current.mean_dimension('x') - 1

    return params.loss_function(residual)
