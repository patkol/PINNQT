import torch

from kolpinn.mathematics import complex_abs2, grad
from kolpinn.grid_quantities import get_fd_second_derivative, mean_dimension, restrict

import parameters as params
import physics


def SE_loss_trafo(qs, *, qs_full, with_grad, i, N):
    """
    i: layer index in [1,N]
    For a constant effective mass only if fd_second_derivatives!
    """

    q = qs[f'bulk{i}']
    q_full = qs_full[f'bulk{i}']
    b_r = qs[f'boundary{N}'][f'b{N+1}']

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
    residual /= b_r
    residual /= physics.V_OOM
    q[f'SE_loss{i}'] = params.loss_function(residual)

    return qs


def j_loss_trafo(qs, *, i, N):
    q = qs[f'bulk{i}']
    b_r = qs[f'boundary{N}'][f'b{N+1}']

    prob_current = torch.imag(physics.H_BAR * torch.conj(q[f'phi{i}'])
                              * q[f'phi{i}_dx'] / q[f'm_eff{i}'])
    residual = prob_current - mean_dimension('x', prob_current, q.grid)
    residual /= complex_abs2(b_r)
    residual /= physics.PROBABILITY_CURRENT_OOM
    q[f'j_loss{i}'] = params.loss_function(residual)

    return qs