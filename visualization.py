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
    N = device.n_layers

    path_prefix = None
    for trainer in device.trainers.values():
        trainer_path_prefix = f'plots/{trainer.saved_parameters_index:04d}/'
        if path_prefix is None:
            path_prefix = trainer_path_prefix
        else:
            assert path_prefix == trainer_path_prefix

        visualization.save_training_history_plot(trainer, path_prefix)


    qs = device.get_extended_qs()


    # Loss vs. E plots

    for grid_name, loss_names in device.used_losses.items():
        q = qs[grid_name]
        for loss_name in loss_names:
            save_lineplot(
                q[loss_name],
                q.grid,
                loss_name,
                'DeltaE',
                'voltage',
                path_prefix = path_prefix,
                x_unit = physics.EV, x_unit_name = 'eV',
                lines_unit = physics.VOLT, lines_unit_name = 'V',
            )

    for i in range(1,N+1):
        q = qs[f'bulk{i}']
        save_lineplot(
            q[f'n{i}'],
            q.grid,
            f'n{i}',
            'x',
            'voltage',
            path_prefix = path_prefix,
            quantity_unit = 1/physics.NM,
            quantity_unit_name = '1/nm',
            x_unit = physics.NM, x_unit_name = 'nm',
            lines_unit = physics.VOLT, lines_unit_name = 'V',
        )


    # Per voltage plots

    voltages = next(iter(qs.values())).grid['voltage']
    for voltage_index, voltage in enumerate(voltages):
        if voltage_index % params.plot_each_voltage != 0:
            continue

        voltage_path_prefix = f'{path_prefix}{voltage:.2f}V/'
        voltage_index_dict = {'voltage': [voltage_index]}

        for contact in device.contacts:
            q_in = qs[contact.grid_name]
            in_boundary_grid = Subgrid(q_in.grid, voltage_index_dict, copy_all=False)
            q_in = restrict_quantities(q_in, in_boundary_grid)
            q_out = qs[f'boundary{contact.out_boundary_index}']
            out_boundary_grid = Subgrid(q_out.grid, voltage_index_dict, copy_all=False)
            q_out = restrict_quantities(q_out, out_boundary_grid)

            incoming_coeff_in = q_in[f'{contact.incoming_coeff_in_name}{contact.index}_{contact}']
            outgoing_coeff_in = q_in[f'{contact.outgoing_coeff_in_name}{contact.index}_{contact}']
            outgoing_coeff_out = q_out[f'{contact.outgoing_coeff_out_name}{contact.out_index}_propagated_{contact}']

            # Layers
            for i in range(1,N+1):
                q = qs[f'bulk{i}']
                bulk_grid = Subgrid(q.grid, voltage_index_dict, copy_all=False)
                q = restrict_quantities(q, bulk_grid)

                ## Wave function
                save_lineplot(
                    complex_abs2(q[f'phi{i}_{contact}']),
                    q.grid,
                    f'|phi{i}_{contact}|^2',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    x_unit_name = 'nm',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )

                fig, ax = plt.subplots()
                add_lineplot(
                    ax,
                    torch.real(q[f'a_output{i}_{contact}']),
                    q.grid,
                    f'Re(a_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'blue',
                )
                add_lineplot(
                    ax,
                    torch.imag(q[f'a_output{i}_{contact}']),
                    q.grid,
                    f'Im(a_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'blue',
                    linestyle = 'dashed',
                )
                add_lineplot(
                    ax,
                    torch.real(q[f'b_output{i}_{contact}']),
                    q.grid,
                    f'Re(b_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'red',
                )
                add_lineplot(
                    ax,
                    torch.imag(q[f'b_output{i}_{contact}']),
                    q.grid,
                    f'Im(b_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'red',
                    linestyle = 'dashed',
                )
                ax.set_xlabel('x [nm]')
                ax.grid(visible=True)
                fig.savefig(voltage_path_prefix + f'outputs{i}_{contact}.pdf')
                plt.close(fig)

                save_lineplot(
                    q[f'j{i}_{contact}'],
                    q.grid,
                    f'prob_current{i}_{contact}',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    x_unit_name = 'nm',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    quantity_unit = physics.H_BAR / physics.M_E / physics.NM,
                    quantity_unit_name = 'hbar/m0/nm',
                    path_prefix = voltage_path_prefix,
                )

                save_heatmap(
                    q[f'DOS{i}_{contact}'],
                    q.grid,
                    f'DOS{i}_{contact}',
                    'x',
                    'DeltaE',
                    quantity_unit = 1/physics.NM/physics.EV,
                    quantity_unit_name = '1/nm/eV',
                    x_unit = physics.NM, x_unit_name = 'nm',
                    y_unit = physics.EV, y_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )

                if not params.extra_plots:
                    continue

                save_lineplot(
                    torch.real(q[f'phi{i}_{contact}']),
                    q.grid,
                    f'Re(phi{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    x_unit_name = 'nm',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )
                save_lineplot(
                    torch.imag(q[f'phi{i}_{contact}']),
                    q.grid,
                    f'Im(phi{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    x_unit_name = 'nm',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )
                visualization.save_complex_polar_plot(
                    q[f'phi{i}_{contact}'],
                    q.grid,
                    f'phi{i}_{contact}',
                    'x',
                    'DeltaE',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )


                ## a, b
                visualization.save_complex_polar_plot(
                    q[f'a{i}_{contact}'],
                    q.grid,
                    f'a{i}_{contact}',
                    'x',
                    'DeltaE',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )
                visualization.save_complex_polar_plot(
                    q[f'b{i}_{contact}'],
                    q.grid,
                    f'b{i}_{contact}',
                    'x',
                    'DeltaE',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )

                ## Losses
                save_lineplot(
                    q[f'SE_loss{i}_{contact}'],
                    q.grid,
                    f'SE_loss{i}_{contact}',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    x_unit_name = 'nm',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    quantity_unit = physics.H_BAR / physics.M_E / physics.NM,
                    quantity_unit_name = 'hbar/m0/nm',
                    path_prefix = voltage_path_prefix,
                )


            # Transmission and reflection probabilities

            abs_group_velocity_in_contact = torch.sqrt(torch.abs(
                2*(q_in[f'E_{contact}']-q_in[f'V{contact.index}'])
                / q_in[f'm_eff{contact.index}']
            ))
            abs_group_velocity_out_contact = torch.sqrt(torch.abs(
                2*(q_out[f'E_{contact}']-q_out[f'V{contact.out_index}'])
                / q_out[f'm_eff{contact.out_index}']
            ))
            v_ratio = abs_group_velocity_out_contact / abs_group_velocity_in_contact

            fig, ax = plt.subplots()
            add_lineplot(
                ax,
                complex_abs2(outgoing_coeff_in) / complex_abs2(incoming_coeff_in),
                q_in.grid,
                'Reflection probability',
                'DeltaE',
                x_unit=physics.EV,
                marker = 'x',
                linewidth = 0,
                c='blue',
            )
            add_lineplot(
                ax,
                complex_abs2(outgoing_coeff_out) / complex_abs2(incoming_coeff_in) * v_ratio,
                q_out.grid,
                'Transmission probability',
                'DeltaE',
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
                    f'{matlab_path}TE{contact}_{params.simulated_device_name}.txt',
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
            fig.savefig(voltage_path_prefix + f'coefficients_vs_E_{contact}.pdf')
            plt.close(fig)