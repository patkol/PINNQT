# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import os
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import torch

from kolpinn.mathematics import complex_abs2
from kolpinn.grids import Subgrid
from kolpinn.quantities import restrict_quantities
from kolpinn import training
from kolpinn.training import Trainer
from kolpinn import visualization
from kolpinn.visualization import add_lineplot, save_lineplot, save_heatmap

import parameters as params
import physics
from classes import Device


def save_plots(trainer: Trainer, device: Device):
    plt.rcParams.update({"font.size": 22})
    path_prefix = f"plots/{trainer.config.saved_parameters_index:04d}/"

    visualization.save_training_history_plot(trainer, path_prefix)

    qs = training.get_extended_qs(trainer.state)

    # Loss vs. E plots
    for grid_name, loss_names in trainer.config.loss_quantities.items():
        q = qs[grid_name]
        for loss_name in loss_names:
            save_lineplot(
                q[loss_name],
                q.grid,
                loss_name,
                "DeltaE",
                "voltage",
                path_prefix=path_prefix,
                x_unit=physics.EV,
                x_unit_name="eV",
                lines_unit=physics.VOLT,
                lines_unit_name="V",
            )

    q = qs["bulk"]

    save_lineplot(
        q["V_int"],
        q.grid,
        "V_int",
        "x",
        path_prefix=path_prefix,
        quantity_unit=physics.EV,
        quantity_unit_name="eV",
        x_unit=physics.NM,
        x_unit_name="nm",
    )

    save_lineplot(
        q["V_int"] + q["V_el"],
        q.grid,
        "V",
        "x",
        "voltage",
        path_prefix=path_prefix,
        quantity_unit=physics.EV,
        quantity_unit_name="eV",
        x_unit=physics.NM,
        x_unit_name="nm",
        lines_unit=physics.VOLT,
        lines_unit_name="V",
    )

    save_lineplot(
        q["m_eff"],
        q.grid,
        "m_eff",
        "x",
        "voltage",
        path_prefix=path_prefix,
        quantity_unit=1 / physics.NM,
        quantity_unit_name="1/nm",
        x_unit=physics.NM,
        x_unit_name="nm",
        lines_unit=physics.VOLT,
        lines_unit_name="V",
    )

    save_lineplot(
        q["n"],
        q.grid,
        "n",
        "x",
        "voltage",
        path_prefix=path_prefix,
        quantity_unit=1 / physics.NM,
        quantity_unit_name="1/nm",
        x_unit=physics.NM,
        x_unit_name="nm",
        lines_unit=physics.VOLT,
        lines_unit_name="V",
    )

    fig, ax = plt.subplots()
    add_lineplot(
        ax,
        q["I"],
        q.grid,
        "PINN",
        "voltage",
        quantity_unit=1e6 / physics.CM**2,
        # quantity_unit_name='10^6 A/cm^2',
        x_unit=physics.VOLT,
        x_unit_name="V",
        marker=".",
        linewidth=0,
        c="blue",
    )
    try:
        matlab_path = "matlab_results/"
        voltages_matlab = np.loadtxt(
            f"{matlab_path}Vbias_{params.simulated_device_name}.txt",
            delimiter=",",
        )
        currents_matlab = np.loadtxt(
            f"{matlab_path}Id_{params.simulated_device_name}.txt",
            delimiter=",",
        )
        ax.plot(
            voltages_matlab,
            currents_matlab / 1e6,
            label="MATLAB Reference",
            linestyle="dashed",
            c="black",
        )
    except FileNotFoundError:
        pass
    ax.set_xlabel("U [V]")
    ax.set_ylabel("I [10^6 A/cm^2]")
    ax.grid(visible=True)
    ax.legend()
    fig.savefig(path_prefix + "I.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for contact in device.contacts:
        add_lineplot(
            ax,
            q[f"I_{contact}"],
            q.grid,
            f"I_{contact}",
            "voltage",
            quantity_unit=1 / physics.CM**2,
            x_unit=physics.VOLT,
            x_unit_name="V",
        )
        ax.set_xlabel("U [V]")
        ax.set_ylabel("I [A/cm^2]")
        ax.grid(visible=True)
        fig.savefig(path_prefix + "I_components.pdf", bbox_inches="tight")
        plt.close(fig)

    # Per voltage plots
    voltages = next(iter(qs.values())).grid["voltage"]
    used_energy_indices = list(
        range(
            0,
            qs["boundary0"].grid.dim_size["DeltaE"],
            params.plot_each_energy,
        )
    )
    energies_index_dict = {"DeltaE": used_energy_indices}
    for voltage_index, voltage in enumerate(voltages):
        if (
            params.plot_each_voltage <= 0
            or voltage_index % params.plot_each_voltage != 0
        ):
            continue

        voltage_path_prefix = f"{path_prefix}{voltage:.2f}V/"
        os.makedirs(voltage_path_prefix, exist_ok=True)
        voltage_index_dict = {"voltage": [voltage_index]}

        for contact in device.contacts:
            q_full = qs["bulk"]
            full_grid = Subgrid(q_full.grid, voltage_index_dict, copy_all=False)
            q_full = restrict_quantities(q_full, full_grid)
            full_grid_reduced = Subgrid(full_grid, energies_index_dict, copy_all=False)
            q_full_reduced = restrict_quantities(q_full, full_grid_reduced)

            q_in = qs[contact.grid_name]
            in_boundary_grid = Subgrid(q_in.grid, voltage_index_dict, copy_all=False)
            q_in = restrict_quantities(q_in, in_boundary_grid)
            in_boundary_grid_reduced = Subgrid(
                in_boundary_grid, energies_index_dict, copy_all=False
            )
            q_in_reduced = restrict_quantities(q_in, in_boundary_grid_reduced)

            q_out = qs[contact.out_boundary_name]
            out_boundary_grid = Subgrid(q_out.grid, voltage_index_dict, copy_all=False)
            q_out = restrict_quantities(q_out, out_boundary_grid)
            out_boundary_grid_reduced = Subgrid(
                out_boundary_grid, energies_index_dict, copy_all=False
            )
            q_out_reduced = restrict_quantities(q_out, out_boundary_grid_reduced)

            # Wave function
            save_lineplot(
                complex_abs2(q_full_reduced[f"phi_{contact}"]),
                q_full_reduced.grid,
                f"|phi_{contact}|^2",
                "x",
                "DeltaE",
                x_unit=physics.NM,
                x_unit_name="nm",
                lines_unit=physics.EV,
                lines_unit_name="eV",
                path_prefix=voltage_path_prefix,
            )
            save_lineplot(
                torch.real(q_full_reduced[f"phi_{contact}"]),
                q_full_reduced.grid,
                f"Re(phi_{contact})",
                "x",
                "DeltaE",
                x_unit=physics.NM,
                x_unit_name="nm",
                lines_unit=physics.EV,
                lines_unit_name="eV",
                path_prefix=voltage_path_prefix,
            )
            save_lineplot(
                torch.imag(q_full_reduced[f"phi_{contact}"]),
                q_full_reduced.grid,
                f"Im(phi_{contact})",
                "x",
                "DeltaE",
                x_unit=physics.NM,
                x_unit_name="nm",
                lines_unit=physics.EV,
                lines_unit_name="eV",
                path_prefix=voltage_path_prefix,
            )
            visualization.save_complex_polar_plot(
                q_full_reduced[f"phi_{contact}"],
                q_full_reduced.grid,
                f"phi_{contact}",
                "x",
                "DeltaE",
                lines_unit=physics.EV,
                lines_unit_name="eV",
                path_prefix=voltage_path_prefix,
            )
            save_heatmap(
                q_full[f"DOS_{contact}"],
                q_full.grid,
                f"DOS_{contact}",
                "x",
                "DeltaE",
                quantity_unit=1 / physics.NM / physics.EV,
                quantity_unit_name="1/nm/eV",
                x_unit=physics.NM,
                x_unit_name="nm",
                y_unit=physics.EV,
                y_unit_name="eV",
                path_prefix=voltage_path_prefix,
            )

            # Transmission and reflection probabilities
            fig, ax = plt.subplots()
            add_lineplot(
                ax,
                q_full[f"R_{contact}"],
                q_full.grid,
                "Reflection probability",
                "DeltaE",
                x_quantity=q_full[f"E_{contact}"],
                x_unit=physics.EV,
                marker="x",
                linewidth=0,
                c="blue",
            )
            add_lineplot(
                ax,
                q_full[f"T_{contact}"],
                q_full.grid,
                "Transmission probability",
                "DeltaE",
                x_quantity=q_full[f"E_{contact}"],
                x_unit=physics.EV,
                marker="x",
                linewidth=0,
                c="orange",
            )

            try:
                matlab_path = f"matlab_results/{voltage:.2f}V/"
                energies_matlab = np.loadtxt(
                    f"{matlab_path}E_{params.simulated_device_name}.txt",
                    delimiter=",",
                )
                a_r_2_left_matlab = np.loadtxt(
                    f"{matlab_path}TE{contact}_{params.simulated_device_name}.txt",
                    delimiter=",",
                )
                ax.plot(
                    energies_matlab,
                    a_r_2_left_matlab,
                    label="MATLAB Transmission",
                    linestyle="dashed",
                    c="orange",
                )
                ax.plot(
                    energies_matlab,
                    1 - a_r_2_left_matlab,
                    label="1 - MATLAB Transmission",
                    linestyle="dashed",
                    c="blue",
                )
            except FileNotFoundError:
                pass

            ax.set_xlabel("E [eV]")
            ax.set_ylim(bottom=-0.1, top=1.1)
            ax.grid(visible=True)
            fig.savefig(
                voltage_path_prefix + f"coefficients_vs_E_{contact}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

            save_lineplot(
                q_full[f"I_spectrum_{contact}"],
                q_full.grid,
                f"Spectral current {contact}",
                "DeltaE",
                x_quantity=q_full[f"E_{contact}"],
                x_label="E",
                quantity_unit=1 / physics.CM**2 / physics.EV,
                quantity_unit_name="A/cm^2/eV",
                x_unit=physics.EV,
                x_unit_name="eV",
                path_prefix=voltage_path_prefix,
            )

            save_lineplot(
                q_full["n"],
                q_full.grid,
                "n",
                "x",
                quantity_unit=1 / physics.CM**3,
                quantity_unit_name="1/cm$^3$",
                x_unit=1 / physics.NM,
                x_unit_name="nm",
                path_prefix=voltage_path_prefix,
            )

            save_lineplot(
                q["doping"] - q["n"],
                q_full.grid,
                "Total Charge Density",
                "x",
                quantity_unit=1 / physics.CM**3,
                quantity_unit_name="q/cm$^3$",
                x_unit=1 / physics.NM,
                x_unit_name="nm",
                path_prefix=voltage_path_prefix,
            )

            save_lineplot(
                q_full["V_el"],
                q_full.grid,
                "V_el",
                "x",
                quantity_unit=physics.EV,
                quantity_unit_name="eV",
                x_unit=physics.NM,
                x_unit_name="nm",
                path_prefix=voltage_path_prefix,
            )
