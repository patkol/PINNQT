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

    dE_dk_left = torch.sqrt(2 * physics.H_BAR**2
                            * (q_left['E'] - q_left['V0'])
                            / q_left['m_eff0'])

    # Layers
    for i in range(1, device.n_layers+1):
        q = qs['bulk' + str(i)]

        ## Wave function
        phi_i = q['phi'+str(i)]
        save_lineplot(
            complex_abs2(phi_i),
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
            torch.real(phi_i),
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
            torch.imag(phi_i),
            f'Im(phi{i})',
            'x',
            'E',
            x_unit = physics.NM,
            x_unit_name = 'nm',
            lines_unit = physics.EV,
            lines_unit_name = 'eV',
            path_prefix = path_prefix,
        )
        visualization.save_complex_polar_plot(
            phi_i,
            f'phi{i}',
            'x',
            'E',
            lines_unit = physics.EV,
            lines_unit_name = 'eV',
            path_prefix = path_prefix,
        )


        ## a, b
        if params.model_ab:
            visualization.save_complex_polar_plot(
                q[f'a{i}'],
                f'a{i}',
                'x',
                'E',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = path_prefix,
            )
            visualization.save_complex_polar_plot(
                q[f'b{i}'],
                f'b{i}',
                'x',
                'E',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = path_prefix,
            )



        ## Probability current
        prob_current = torch.imag(physics.H_BAR * torch.conj(phi_i)
                                  * q['phi_dx'+str(i)] / q['m_eff'+str(i)])
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
        dos = 1/(2*np.pi) * complex_abs2(phi_i) / dE_dk_left.values
        save_heatmap(
            dos, 'DOS'+str(i), 'x', 'E',
            quantity_unit = 1/physics.NM/physics.EV, quantity_unit_name = '1/nm/eV',
            x_unit = physics.NM, x_unit_name = 'nm',
            y_unit = physics.EV, y_unit_name = 'eV',
            path_prefix=path_prefix,
        )

        ## Losses
        save_lineplot(
            q['SE'+str(i)],
            f'SE{i}',
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

        # NN inputs and outputs
        for label in ('x_transformed', 'nn_output0', 'nn_output1'):
            save_lineplot(
                q[label],
                f'{label}_layer{i}',
                'x',
                'E',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = path_prefix,
            )


    # Transmission and reflection probabilities

    abs_group_velocity_left_contact = torch.sqrt(torch.abs(
        2*(q_left['E']-q_left['V'+left_contact_index])
        / q_left['m_eff'+left_contact_index]
    ))
    abs_group_velocity_right_contact = torch.sqrt(torch.abs(
        2*(q_right['E']-q_right['V'+right_contact_index])
        / q_right['m_eff'+right_contact_index]
    ))
    v_ratio_values = abs_group_velocity_right_contact.values / abs_group_velocity_left_contact.values
    v_ratio = Quantity(v_ratio_values, q_right.grid)

    b_l = q_left['phi'+left_layer_index] - physics.A_L
    b_r = q_right['phi'+right_layer_index] - physics.A_R


    fig, ax = plt.subplots()
    add_lineplot(
        ax,
        complex_abs2(b_l),
        'Reflection probability',
        'E',
        x_unit=physics.EV,
        marker = 'D',
        linewidth = 0,
        c='blue',
    )
    add_lineplot(
        ax,
        v_ratio * complex_abs2(b_r),
        'Transmission probability',
        'E',
        x_unit=physics.EV,
        marker = 'D',
        linewidth = 0,
        c='orange',
    )

    try:
        energies_matlab = np.loadtxt(
            f'matlab_results/E_{params.simulated_device_name}.txt',
            delimiter=',',
        )
        b_r_2_left_matlab = np.loadtxt(
            f'matlab_results/TEL_{params.simulated_device_name}.txt',
            delimiter=',',
        )
        ax.plot(
            energies_matlab,
            b_r_2_left_matlab,
            label='MATLAB Transmission',
            linestyle='dashed',
            c='orange',
        )
        ax.plot(
            energies_matlab,
            1 - b_r_2_left_matlab,
            label='1 - MATLAB Transmission',
            linestyle='dashed',
            c='blue',
        )
    except:
        pass

    ax.set_xlabel('E [eV]')
    ax.set_ylim(bottom=-0.1, top=1.1)
    ax.grid(visible=True)
    ax.legend()
    fig.savefig(path_prefix + 'coefficients_vs_E.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()
    add_lineplot(ax, torch.real(b_l), 'Re(b_l)', 'E', x_unit=physics.EV)
    add_lineplot(ax, torch.imag(b_l), 'Im(b_l)', 'E', x_unit=physics.EV)
    add_lineplot(ax, torch.real(b_r), 'Re(b_r)', 'E', x_unit=physics.EV)
    add_lineplot(ax, torch.imag(b_r), 'Im(b_r)', 'E', x_unit=physics.EV)
    ax.set_xlabel('E [eV]')
    ax.set_ylim(bottom=-1.1, top=1.1)
    ax.grid(visible=True)
    ax.legend()
    fig.savefig(path_prefix + 'coefficients_vs_E_complex.pdf')
    plt.close(fig)
