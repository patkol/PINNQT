import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from kolpinn.mathematics import complex_abs2
from kolpinn.grid_quantities import Quantity
from kolpinn.model import get_extended_q
from kolpinn import visualization
from kolpinn.visualization import add_lineplot, save_lineplot, save_heatmap

import parameters as params
import physics



def visualize(device):
    trainer = device.trainer
    path_prefix = f'plots/{trainer.saved_weights_index:04d}/'
    os.makedirs(path_prefix, exist_ok=True)
    visualization.save_training_history_plot(trainer)
    visualization.save_loss_plots(trainer, 'E')


    # Transmission and reflection probabilities

    left_contact_index = '0'
    left_layer_index = '1'
    right_layer_index = str(device.n_layers)
    right_contact_index = str(device.n_layers+1)

    batcher_left = trainer.batchers_validation['boundary'+left_contact_index]
    batcher_right = trainer.batchers_validation['boundary'+right_layer_index]
    q_left = batcher_left.get_extended_q(
        trainer.models,
        trainer.model_parameters,
        trainer.diffable_quantities,
    )
    q_right = batcher_right.get_extended_q(
        trainer.models,
        trainer.model_parameters,
        trainer.diffable_quantities,
    )

    abs_group_velocity_left_contact = (2*(q_left['E']-q_left['V'+left_contact_index])
                                       / q_left['m_eff'+left_contact_index]).transform(torch.abs).transform(torch.sqrt)
    abs_group_velocity_right_contact = (2*(q_right['E']-q_right['V'+right_contact_index])
                                        / q_right['m_eff'+right_contact_index]).transform(torch.abs).transform(torch.sqrt)
    v_ratio_values = abs_group_velocity_right_contact.values / abs_group_velocity_left_contact.values
    v_ratio = Quantity(v_ratio_values, q_right.grid)

    b_l = q_left['phi'+left_layer_index] - physics.A_L
    b_r = q_right['phi'+left_layer_index] - physics.A_R

    fig, ax = plt.subplots()
    add_lineplot(ax, b_l.transform(complex_abs2), 'Reflection probability', 'E', x_unit=physics.EV)
    add_lineplot(ax, v_ratio * b_r.transform(complex_abs2), 'Transmission probability', 'E', x_unit=physics.EV)
    ax.set_xlabel('E [eV]')
    ax.set_ylim(bottom=-0.1, top=1.1)
    ax.grid(visible=True)
    ax.legend()
    fig.savefig(path_prefix + 'coefficients_vs_E.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()
    add_lineplot(ax, b_l.transform(torch.real), 'Re(b_l)', 'E', x_unit=physics.EV)
    add_lineplot(ax, b_l.transform(torch.imag), 'Im(b_l)', 'E', x_unit=physics.EV)
    add_lineplot(ax, b_r.transform(torch.real), 'Re(b_r)', 'E', x_unit=physics.EV)
    add_lineplot(ax, b_r.transform(torch.imag), 'Im(b_r)', 'E', x_unit=physics.EV)
    ax.set_xlabel('E [eV]')
    ax.set_ylim(bottom=-1.1, top=1.1)
    ax.grid(visible=True)
    ax.legend()
    fig.savefig(path_prefix + 'coefficients_vs_E_complex.pdf')
    plt.close(fig)


    # Bulk: Wave functions and DOS

    dE_dk_left_squared = (2 * physics.H_BAR**2 * (q_left['E'] - q_left['V0'])
                          / q_left['m_eff0'])
    dE_dk_left = dE_dk_left_squared.transform(torch.sqrt)

    for i in range(1, device.n_layers+1):
        batcher = trainer.batchers_validation['bulk'+str(i)]
        # OPTIM: Only evaluate phi[i] here
        diffable_quantities = {'x_expanded': lambda q: q['x'],}
        diffable_quantities.update(trainer.diffable_quantities)

        q = get_extended_q(
            batcher.q_full,
            models = trainer.models,
            models_require_grad = False,
            model_parameters = trainer.model_parameters,
            diffable_quantities = diffable_quantities,
            quantities_requiring_grad_labels = ['x'],
        )


        ## Wave functions

        save_lineplot(
            q['phi'+str(i)].transform(complex_abs2),
            f'|phi{i}|^2',
            'x',
            'E',
            x_unit = physics.NM,
            x_unit_name = 'nm',
            lines_unit = physics.EV,
            lines_unit_name = 'eV',
            path_prefix = path_prefix,
        )
        save_lineplot(
            q['phi'+str(i)].transform(torch.real),
            f'Re(phi{i})',
            'x',
            'E',
            x_unit = physics.NM,
            x_unit_name = 'nm',
            lines_unit = physics.EV,
            lines_unit_name = 'eV',
            path_prefix = path_prefix,
        )
        save_lineplot(
            q['phi'+str(i)].transform(torch.imag),
            f'Im(phi{i})',
            'x',
            'E',
            x_unit = physics.NM,
            x_unit_name = 'nm',
            lines_unit = physics.EV,
            lines_unit_name = 'eV',
            path_prefix = path_prefix,
        )

        phi = q['phi'+str(i)]
        phi_dx = phi.get_grad(
            q['x_expanded'],
            retain_graph=True, # Necessary for evaluation in later layers
            create_graph=False,
        )
        phi_dx_abs2 = phi_dx.transform(complex_abs2)
        k_abs2 = (2 * (q['E']-q['V'+str(i)]) * q['m_eff'+str(i)]
                  / physics.H_BAR**2).transform(torch.abs)
        prob_current_complex = physics.H_BAR * phi.transform(torch.conj) * phi_dx / q['m_eff'+str(i)]
        save_lineplot(
            prob_current_complex.transform(torch.real),
            f"Re(prob_current{i})",
            'x',
            'E',
            x_unit = physics.NM,
            x_unit_name = 'nm',
            lines_unit = physics.EV,
            lines_unit_name = 'eV',
            quantity_unit = physics.H_BAR / physics.M_E / physics.NM,
            quantity_unit_name = 'hbar/m0/nm',
            path_prefix = path_prefix,
        )
        save_lineplot(
            prob_current_complex.transform(torch.imag),
            f"Im(prob_current{i})",
            'x',
            'E',
            x_unit = physics.NM,
            x_unit_name = 'nm',
            lines_unit = physics.EV,
            lines_unit_name = 'eV',
            quantity_unit = physics.H_BAR / physics.M_E / physics.NM,
            quantity_unit_name = 'hbar/m0/nm',
            path_prefix = path_prefix,
        )




        ## DOS

        dE_dk_left = Quantity(dE_dk_left.values, q.grid)
        dos = 1/(2*np.pi) * q['phi'+str(i)].transform(complex_abs2) / dE_dk_left
        save_heatmap(
            dos, 'DOS'+str(i), 'x', 'E',
            quantity_unit = 1/physics.NM/physics.EV, quantity_unit_name = '1/nm/eV',
            x_unit = physics.NM, x_unit_name = 'nm',
            y_unit = physics.EV, y_unit_name = 'eV',
            path_prefix=path_prefix,
        )
        dos_inverted = dos.transform(lambda t: torch.flip(t, [1]))
        dos_symmetrized = dos + dos_inverted
        save_heatmap(
            dos_symmetrized, 'DOS'+str(i)+'_symmetrized', 'x', 'E',
            quantity_unit = 1/physics.NM/physics.EV, quantity_unit_name = '1/nm/eV',
            x_unit = physics.NM, x_unit_name = 'nm',
            y_unit = physics.EV, y_unit_name = 'eV',
            path_prefix=path_prefix,
        )


