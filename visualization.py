# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import os
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import torch

from kolpinn.mathematics import complex_abs2
from kolpinn.grid_quantities import Subgrid, restrict_quantities
from kolpinn import visualization
from kolpinn.visualization import add_lineplot, save_lineplot, save_heatmap

import parameters as params
import physics



def visualize(device):
    plt.rcParams.update({'font.size': 22})

    N = device.n_layers

    path_prefix = f'plots/{device.trainer.saved_parameters_index:04d}/'

    visualization.save_training_history_plot(device.trainer, path_prefix)


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

    q = qs['full']
    save_lineplot(
        q[f'n'],
        q.grid,
        f'n',
        'x',
        'voltage',
        path_prefix = path_prefix,
        quantity_unit = 1/physics.NM,
        quantity_unit_name = '1/nm',
        x_unit = physics.NM, x_unit_name = 'nm',
        lines_unit = physics.VOLT, lines_unit_name = 'V',
    )

    fig, ax = plt.subplots()
    add_lineplot(
        ax,
        q['I'],
        q.grid,
        'PINN',
        'voltage',
        quantity_unit=1e6 / physics.CM**2,
        # quantity_unit_name='10^6 A/cm^2',
        x_unit=physics.VOLT, x_unit_name='V',
        marker='.',
        linewidth=0,
        c='blue',
    )
    try:
        matlab_path = 'matlab_results/'
        voltages_matlab = np.loadtxt(
            f'{matlab_path}Vbias_{params.simulated_device_name}.txt',
            delimiter=',',
        )
        currents_matlab = np.loadtxt(
            f'{matlab_path}Id_{params.simulated_device_name}.txt',
            delimiter=',',
        )
        ax.plot(
            voltages_matlab,
            currents_matlab / 1e6,
            label='MATLAB Reference',
            linestyle='dashed',
            c='black',
        )
    except:
        pass
    ax.set_xlabel('U [V]')
    ax.set_ylabel('I [10^6 A/cm^2]')
    ax.grid(visible=True)
    ax.legend()
    fig.savefig(path_prefix + 'I.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    for contact in device.contacts:
        add_lineplot(
            ax,
            q[f'I_{contact}'],
            q.grid,
            f'I_{contact}',
            'voltage',
            quantity_unit = 1 / physics.CM**2,
            x_unit = physics.VOLT, x_unit_name = 'V',
        )
        ax.set_xlabel('U [V]')
        ax.set_ylabel('I [A/cm^2]')
        ax.grid(visible=True)
        fig.savefig(path_prefix + f'I_components.pdf', bbox_inches='tight')
        plt.close(fig)


    # Per voltage plots

    voltages = next(iter(qs.values())).grid['voltage']
    used_energy_indices = list(range(
        0,
        qs['boundary0'].grid.dim_size['DeltaE'],
        params.plot_each_energy,
    ))
    energies_index_dict = {'DeltaE': used_energy_indices}
    for voltage_index, voltage in enumerate(voltages):
        if params.plot_each_voltage <= 0 or voltage_index % params.plot_each_voltage != 0:
            continue

        voltage_path_prefix = f'{path_prefix}{voltage:.2f}V/'
        os.makedirs(voltage_path_prefix, exist_ok=True)
        voltage_index_dict = {'voltage': [voltage_index]}

        for contact in device.contacts:
            q_full = qs['full']
            full_grid = Subgrid(q_full.grid, voltage_index_dict, copy_all=False)
            q_full = restrict_quantities(q_full, full_grid)
            full_grid_reduced = Subgrid(full_grid, energies_index_dict, copy_all=False)
            q_full_reduced = restrict_quantities(q_full, full_grid_reduced)

            q_in = qs[contact.grid_name]
            in_boundary_grid = Subgrid(q_in.grid, voltage_index_dict, copy_all=False)
            q_in = restrict_quantities(q_in, in_boundary_grid)
            in_boundary_grid_reduced = Subgrid(in_boundary_grid, energies_index_dict, copy_all=False)
            q_in_reduced = restrict_quantities(q_in, in_boundary_grid_reduced)

            q_out = qs[contact.out_boundary_name]
            out_boundary_grid = Subgrid(q_out.grid, voltage_index_dict, copy_all=False)
            q_out = restrict_quantities(q_out, out_boundary_grid)
            out_boundary_grid_reduced = Subgrid(out_boundary_grid, energies_index_dict, copy_all=False)
            q_out_reduced = restrict_quantities(q_out, out_boundary_grid_reduced)

            # Layers
            for i in range(1,N+1):
                q = qs[f'bulk{i}']
                bulk_grid = Subgrid(q.grid, voltage_index_dict, copy_all=False)
                q = restrict_quantities(q, bulk_grid)
                bulk_grid_reduced = Subgrid(bulk_grid, energies_index_dict, copy_all=False)
                q_reduced = restrict_quantities(q, bulk_grid_reduced)

                fig, ax = plt.subplots()
                add_lineplot(
                    ax,
                    torch.real(q_reduced[f'a_output{i}_{contact}']),
                    q_reduced.grid,
                    f'Re(a_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'blue',
                )
                add_lineplot(
                    ax,
                    torch.imag(q_reduced[f'a_output{i}_{contact}']),
                    q_reduced.grid,
                    f'Im(a_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'blue',
                    linestyle = 'dashed',
                )
                add_lineplot(
                    ax,
                    torch.real(q_reduced[f'b_output{i}_{contact}']),
                    q_reduced.grid,
                    f'Re(b_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'red',
                )
                add_lineplot(
                    ax,
                    torch.imag(q_reduced[f'b_output{i}_{contact}']),
                    q_reduced.grid,
                    f'Im(b_output{i}_{contact})',
                    'x',
                    'DeltaE',
                    x_unit = physics.NM,
                    c = 'red',
                    linestyle = 'dashed',
                )
                ax.set_xlabel('x [nm]')
                ax.grid(visible=True)
                fig.savefig(
                    voltage_path_prefix + f'outputs{i}_{contact}.pdf',
                    bbox_inches='tight',
                )
                plt.close(fig)

                save_lineplot(
                    q_reduced[f'j{i}_{contact}'],
                    q_reduced.grid,
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

                if not params.extra_plots:
                    continue

                ## a, b
                visualization.save_complex_polar_plot(
                    q_reduced[f'a{i}_{contact}'],
                    q_reduced.grid,
                    f'a{i}_{contact}',
                    'x',
                    'DeltaE',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )
                visualization.save_complex_polar_plot(
                    q_reduced[f'b{i}_{contact}'],
                    q_reduced.grid,
                    f'b{i}_{contact}',
                    'x',
                    'DeltaE',
                    lines_unit = physics.EV,
                    lines_unit_name = 'eV',
                    path_prefix = voltage_path_prefix,
                )

                ## Losses
                save_lineplot(
                    q_reduced[f'SE_loss{i}_{contact}'],
                    q_reduced.grid,
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


            ## Wave function
            save_lineplot(
                complex_abs2(q_full_reduced[f'phi_{contact}']),
                q_full_reduced.grid,
                f'|phi_{contact}|^2',
                'x',
                'DeltaE',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            save_lineplot(
                torch.real(q_full_reduced[f'phi_{contact}']),
                q_full_reduced.grid,
                f'Re(phi_{contact})',
                'x',
                'DeltaE',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            save_lineplot(
                torch.imag(q_full_reduced[f'phi_{contact}']),
                q_full_reduced.grid,
                f'Im(phi_{contact})',
                'x',
                'DeltaE',
                x_unit = physics.NM,
                x_unit_name = 'nm',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            visualization.save_complex_polar_plot(
                q_full_reduced[f'phi_{contact}'],
                q_full_reduced.grid,
                f'phi_{contact}',
                'x',
                'DeltaE',
                lines_unit = physics.EV,
                lines_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
            save_heatmap(
                q_full[f'DOS_{contact}'],
                q_full.grid,
                f'DOS_{contact}',
                'x',
                'DeltaE',
                quantity_unit = 1/physics.NM/physics.EV,
                quantity_unit_name = '1/nm/eV',
                x_unit = physics.NM, x_unit_name = 'nm',
                y_unit = physics.EV, y_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )


            # Transmission and reflection probabilities

            fig, ax = plt.subplots()
            add_lineplot(
                ax,
                q_full[f'R_{contact}'],
                q_full.grid,
                'Reflection probability',
                'DeltaE',
                x_quantity = q_full[f'E_{contact}'],
                x_unit=physics.EV,
                marker = 'x',
                linewidth = 0,
                c='blue',
            )
            add_lineplot(
                ax,
                q_full[f'T_{contact}'],
                q_full.grid,
                'Transmission probability',
                'DeltaE',
                x_quantity = q_full[f'E_{contact}'],
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
            fig.savefig(
                voltage_path_prefix + f'coefficients_vs_E_{contact}.pdf',
                bbox_inches='tight',
            )
            plt.close(fig)


            save_lineplot(
                q_full[f'I_spectrum_{contact}'],
                q_full.grid,
                f'Spectral current {contact}',
                'DeltaE',
                x_quantity = q_full[f'E_{contact}'],
                x_label = 'E',
                quantity_unit = 1 / physics.CM**2 / physics.EV,
                quantity_unit_name = 'A/cm^2/eV',
                x_unit = physics.EV, x_unit_name = 'eV',
                path_prefix = voltage_path_prefix,
            )
