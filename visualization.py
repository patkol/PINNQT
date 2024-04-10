import numpy as np
import matplotlib.pyplot as plt
import torch

from kolpinn.mathematics import complex_abs2
from kolpinn.grid_quantities import Subgrid, restrict_quantities
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


    # Loss vs. E plots

    for grid_name, loss_names in device.used_losses.items():
        q = qs[grid_name]
        for loss_name in loss_names:
            save_lineplot(
                q[loss_name],
                q.grid,
                loss_name,
                'E',
                'voltage',
                path_prefix = path_prefix,
                x_unit = physics.EV,
                x_unit_name = 'eV',
                lines_unit_name = 'V',
            )


    # Per voltage plots

    voltages = next(iter(qs.values())).grid['voltage']
    for voltage_index, voltage in enumerate(voltages):
        voltage_path_prefix = f'{path_prefix}{voltage:.2f}V/'
        voltage_index_dict = {'voltage': [voltage_index]}

        q_left = qs['boundary'+left_boundary_index]
        left_boundary_grid = Subgrid(q_left.grid, voltage_index_dict, copy_all=False)
        q_left = restrict_quantities(q_left, left_boundary_grid)
        q_right = qs['boundary'+right_boundary_index]
        right_boundary_grid = Subgrid(q_right.grid, voltage_index_dict, copy_all=False)
        q_right = restrict_quantities(q_right, right_boundary_grid)

        dE_dk_left = torch.sqrt(2 * physics.H_BAR**2
                                * (q_left['E'] - q_left['V0'])
                                / q_left['m_eff0'])

        a_l = q_left['a0_propagated']
        b_l = q_left['b0_propagated']
        a_r = q_right[f'a{right_layer_index}']
        b_r = q_right[f'b{right_layer_index}']

        # Layers
        for i in range(1, device.n_layers+1):
            q = qs['bulk' + str(i)]
            bulk_grid = Subgrid(q.grid, voltage_index_dict, copy_all=False)
            q = restrict_quantities(q, bulk_grid)

            ## Wave function
            save_lineplot(
                complex_abs2(q[f'phi{i}']),
                q.grid,
                f'|phi{i}|^2',
                'x',
                'E',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            save_lineplot(
                torch.real(q[f'phi{i}']),
                q.grid,
                f'Re(phi{i})',
                'x',
                'E',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            save_lineplot(
                torch.imag(q[f'phi{i}']),
                q.grid,
                f'Im(phi{i})',
                'x',
                'E',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            visualization.save_complex_polar_plot(
                q[f'phi{i}'],
                q.grid,
                f'phi{i}',
                'x',
                'E',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )


            ## a, b
            visualization.save_complex_polar_plot(
                q[f'a{i}'],
                q.grid,
                f'a{i}',
                'x',
                'E',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            visualization.save_complex_polar_plot(
                q[f'b{i}'],
                q.grid,
                f'b{i}',
                'x',
                'E',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )

            fig, ax = plt.subplots()
            add_lineplot(
                ax,
                torch.real(q[f'a_output_untransformed{i}']),
                q.grid,
                f'Re(a_output_untransformed{i})',
                'x',
                'E',
                x_unit = physics.NM,
                c = 'blue',
            )
            add_lineplot(
                ax,
                torch.imag(q[f'a_output_untransformed{i}']),
                q.grid,
                f'Im(a_output_untransformed{i})',
                'x',
                'E',
                x_unit = physics.NM,
                c = 'blue',
                linestyle = 'dashed',
            )
            add_lineplot(
                ax,
                torch.real(q[f'b_output_untransformed{i}']),
                q.grid,
                f'Re(b_output_untransformed{i})',
                'x',
                'E',
                x_unit = physics.NM,
                c = 'red',
            )
            add_lineplot(
                ax,
                torch.imag(q[f'b_output_untransformed{i}']),
                q.grid,
                f'Im(b_output_untransformed{i})',
                'x',
                'E',
                x_unit = physics.NM,
                c = 'red',
                linestyle = 'dashed',
            )
            ax.set_xlabel('x [nm]')
            ax.grid(visible=True)
            fig.savefig(voltage_path_prefix + f'untransformed_outputs{i}.pdf')
            plt.close(fig)


            save_lineplot(
                q[f'b_oom{i}'],
                q.grid,
                f'b_oom{i}',
                'x',
                'E',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )


            ## Probability current
            prob_current = torch.imag(physics.H_BAR * torch.conj(q[f'phi{i}'])
                                      * q[f'phi{i}_dx'] / q[f'm_eff{i}'])
            save_lineplot(
                prob_current,
                q.grid,
                f'prob_current{i}',
                'x',
                'E',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                quantity_unit = physics.H_BAR / physics.M_E / physics.NM,
                quantity_unit_name = 'hbar/m0/nm',
                path_prefix = voltage_path_prefix,
            )

            ## DOS
            dos = 1/(2*np.pi) * complex_abs2(q[f'phi{i}'] / b_r) / dE_dk_left
            save_heatmap(
                dos,
                q.grid,
                f'DOS{i}',
                'x',
                'E',
                q.grid,
                quantity_unit = 1/physics.NM/physics.EV,
                quantity_unit_name = '1/nm/eV',
                x_unit = physics.NM, x_unit_name = 'nm',
                y_unit = physics.EV, y_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )

            ## Losses
            save_lineplot(
                q[f'SE_loss{i}'],
                q.grid,
                f'SE_loss{i}',
                'x',
                'E',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                quantity_unit = physics.H_BAR / physics.M_E / physics.NM,
                quantity_unit_name = 'hbar/m0/nm',
                path_prefix = voltage_path_prefix,
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
        v_ratio = abs_group_velocity_right_contact / abs_group_velocity_left_contact

        fig, ax = plt.subplots()
        add_lineplot(
            ax,
            complex_abs2(a_r) / complex_abs2(b_r),
            q_left.grid,
            'Reflection probability',
            'E',
            x_unit=physics.EV,
            marker = 'x',
            linewidth = 0,
            c='blue',
        )
        add_lineplot(
            ax,
            complex_abs2(b_l) / complex_abs2(b_r) / v_ratio,
            q_right.grid,
            'Transmission probability',
            'E',
            x_unit=physics.EV,
            marker = 'x',
            linewidth = 0,
            c='orange',
        )

        try:
            matlab_path = f'matlab_results/{voltage:.2f}V/'
            energies_matlab = np.loadtxt(
                    f'{matlab_path}E_{params.simulated_device_name}.txt',
                delimiter=',',
            )
            a_r_2_left_matlab = np.loadtxt(
                f'{matlab_path}TEL_{params.simulated_device_name}.txt',
                delimiter=',',
            )
            ax.plot(
                energies_matlab,
                a_r_2_left_matlab,
                label='MATLAB Transmission',
                linestyle='dashed',
                c='orange',
            )
            ax.plot(
                energies_matlab,
                1 - a_r_2_left_matlab,
                label='1 - MATLAB Transmission',
                linestyle='dashed',
                c='blue',
            )
        except:
            pass

        ax.set_xlabel('E [eV]')
        ax.set_ylim(bottom=-0.1, top=1.1)
        ax.grid(visible=True)
        #ax.legend()
        fig.savefig(voltage_path_prefix + 'coefficients_vs_E.pdf')
        plt.close(fig)