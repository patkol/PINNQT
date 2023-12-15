import numpy as np
import matplotlib.pyplot as plt
import torch

from kolpinn.mathematics import complex_abs2
from kolpinn.grid_quantities import Quantity
from kolpinn.model import get_extended_q, get_extended_q_batchwise
from kolpinn import visualization
from kolpinn.visualization import add_lineplot, save_lineplot, save_heatmap

import parameters as params
import physics



def visualize(device):
    path_prefix = None
    for energy_string, trainer in device.trainers.items():
        trainer_path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
        if path_prefix is None:
            path_prefix = trainer_path_prefix
        else:
            assert path_prefix == trainer_path_prefix

        visualization.save_training_history_plot(trainer, path_prefix)


    qs = device.get_extended_qs()

    left_contact_index = '0'
    left_layer_index = '1'
    right_layer_index = str(device.n_layers)
    right_contact_index = str(device.n_layers+1)
    left_boundary_index = '0'
    right_boundary_index = str(device.n_layers)

    q_left = qs['boundary'+left_boundary_index]
    q_right = qs['boundary'+right_boundary_index]

    dE_dk_left = (2 * physics.H_BAR**2
                  * (q_left['E'] - q_left['V0'])
                  / q_left['m_eff0']).transform(torch.sqrt)

    # Layers
    for i in range(1, device.n_layers+1):
        q = qs['bulk' + str(i)]

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

        # Probability current
        prob_current = (physics.H_BAR * q['phi'+str(i)].transform(torch.conj)
                        * q['phi_dx'+str(i)] / q['m_eff'+str(i)]).transform(torch.imag)
        save_lineplot(
            prob_current,
            f"prob_current{i}",
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
        dE_dk_left_regridded = Quantity(dE_dk_left.values, q.grid)
        dos = 1/(2*np.pi) * q['phi'+str(i)].transform(complex_abs2) / dE_dk_left_regridded
        save_heatmap(
            dos, 'DOS'+str(i), 'x', 'E',
            quantity_unit = 1/physics.NM/physics.EV, quantity_unit_name = '1/nm/eV',
            x_unit = physics.NM, x_unit_name = 'nm',
            y_unit = physics.EV, y_unit_name = 'eV',
            path_prefix=path_prefix,
        )


    # Transmission and reflection probabilities

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

    energies_matlab = np.loadtxt('matlab_results/E.txt', delimiter=',')
    b_r_2_left_matlab = np.loadtxt('matlab_results/TEL.txt', delimiter=',')
    ax.plot(energies_matlab, b_r_2_left_matlab, label='MATLAB |b_r|^2', linestyle='dashed', c='green')

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
