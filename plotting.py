# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from typing import Dict
import os
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import torch

from kolpinn.mathematics import complex_abs2
from kolpinn.grids import Subgrid
from kolpinn.quantities import restrict_quantities, QuantityDict
from kolpinn.training import Trainer
from kolpinn import visualization
from kolpinn.visualization import add_lineplot, save_lineplot, save_heatmap

from classes import Device
import physical_constants as consts
import parameters as params


def save_plots(
    qs: Dict[str, QuantityDict],
    trainer: Trainer,
    device: Device,
    *,
    prefix="",
):
    plt.rcParams.update({"font.size": 22})
    path_prefix = f"plots/{trainer.config.saved_parameters_index:04d}/{prefix}"

    visualization.save_training_history_plot(trainer, path_prefix)

    # Loss vs. E plots
    for grid_name, loss_names in trainer.config.loss_quantities.items():
        q = qs[grid_name]
        total_loss = 0
        for loss_name in loss_names:
            save_lineplot(
                q[loss_name],
                q.grid,
                loss_name,
                "DeltaE",
                "voltage",
                avgd_dimensions=("x", "voltage2"),
                path_prefix=path_prefix,
                x_unit=consts.EV,
                x_unit_name="eV",
                lines_unit=consts.VOLT,
                lines_unit_name="V",
            )
            total_loss += q[loss_name]
        save_lineplot(
            total_loss,
            q.grid,
            f"Total_loss_{grid_name}",
            "DeltaE",
            "voltage",
            avgd_dimensions=("x", "voltage2"),
            path_prefix=path_prefix,
            x_unit=consts.EV,
            x_unit_name="eV",
            lines_unit=consts.VOLT,
            lines_unit_name="V",
        )
        save_lineplot(
            total_loss,
            q.grid,
            f"Total_loss_{grid_name}",
            "x",
            "voltage",
            avgd_dimensions=("DeltaE", "voltage2"),
            path_prefix=path_prefix,
            x_unit=consts.NM,
            x_unit_name="nm",
            lines_unit=consts.VOLT,
            lines_unit_name="V",
        )

    q = qs["bulk"]

    save_lineplot(
        q["V_int"],
        q.grid,
        "V_int",
        "x",
        path_prefix=path_prefix,
        quantity_unit=consts.EV,
        quantity_unit_name="eV",
        x_unit=consts.NM,
        x_unit_name="nm",
    )

    save_lineplot(
        -q["I"],
        q.grid,
        "-I",
        "voltage",
        "voltage2" if params.use_voltage2 else None,
        path_prefix=path_prefix,
        quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
        quantity_unit_name="10^6 A/cm^2",
        x_unit=consts.VOLT,
        x_unit_name="V",
        lines_unit=consts.VOLT,
        lines_unit_name="V",
    )
    save_lineplot(
        -q["I_averaged"],
        q.grid,
        "-I_averaged",
        "voltage",
        "voltage2" if params.use_voltage2 else None,
        path_prefix=path_prefix,
        quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
        quantity_unit_name="10^6 A/cm^2",
        x_unit=consts.VOLT,
        x_unit_name="V",
        lines_unit=consts.VOLT,
        lines_unit_name="V",
    )

    if params.use_voltage2:
        save_lineplot(
            -q["I"],
            q.grid,
            "-I",
            "voltage2",
            "voltage",
            path_prefix=path_prefix,
            quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
            quantity_unit_name="10^6 A/cm^2",
            x_unit=consts.VOLT,
            x_unit_name="V",
            lines_unit=consts.VOLT,
            lines_unit_name="V",
        )
        save_lineplot(
            -q["I_averaged"],
            q.grid,
            "-I_averaged",
            "voltage2",
            "voltage",
            path_prefix=path_prefix,
            quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
            quantity_unit_name="10^6 A/cm^2",
            x_unit=consts.VOLT,
            x_unit_name="V",
            lines_unit=consts.VOLT,
            lines_unit_name="V",
        )
    else:
        save_lineplot(
            -q["I_xdep"],
            q.grid,
            "-I_xdep",
            "x",
            "voltage",
            path_prefix=path_prefix,
            quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
            quantity_unit_name="10^6 A/cm^2",
            x_unit=consts.NM,
            x_unit_name="nm",
            lines_unit=consts.VOLT,
            lines_unit_name="V",
        )

    fig, ax = plt.subplots()
    for contact in device.contacts:
        save_lineplot(
            -q[f"I_{contact}"],
            q.grid,
            f"-I_{contact}",
            "voltage",
            "voltage2" if params.use_voltage2 else None,
            path_prefix=path_prefix,
            quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
            quantity_unit_name="10^6 A/cm^2",
            x_unit=consts.VOLT,
            x_unit_name="V",
            lines_unit=consts.VOLT,
            lines_unit_name="V",
        )
        save_lineplot(
            -q[f"I_averaged_{contact}"],
            q.grid,
            f"-I_averaged_{contact}",
            "voltage",
            "voltage2" if params.use_voltage2 else None,
            path_prefix=path_prefix,
            quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
            quantity_unit_name="10^6 A/cm^2",
            x_unit=consts.VOLT,
            x_unit_name="V",
            lines_unit=consts.VOLT,
            lines_unit_name="V",
        )

        if params.use_voltage2:
            save_lineplot(
                -q[f"I_{contact}"],
                q.grid,
                f"-I_{contact}",
                "voltage2",
                "voltage",
                path_prefix=path_prefix,
                quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
                quantity_unit_name="10^6 A/cm^2",
                x_unit=consts.VOLT,
                x_unit_name="V",
                lines_unit=consts.VOLT,
                lines_unit_name="V",
            )
            save_lineplot(
                -q[f"I_averaged_{contact}"],
                q.grid,
                f"-I_averaged_{contact}",
                "voltage2",
                "voltage",
                path_prefix=path_prefix,
                quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
                quantity_unit_name="10^6 A/cm^2",
                x_unit=consts.VOLT,
                x_unit_name="V",
                lines_unit=consts.VOLT,
                lines_unit_name="V",
            )
        else:
            save_lineplot(
                -q[f"I_xdep_{contact}"],
                q.grid,
                f"-I_xdep_{contact}",
                "x",
                "voltage",
                path_prefix=path_prefix,
                quantity_unit=1e6 * consts.COULOMB / consts.SECOND / consts.CM**2,
                quantity_unit_name="10^6 A/cm^2",
                x_unit=consts.NM,
                x_unit_name="nm",
                lines_unit=consts.VOLT,
                lines_unit_name="V",
            )

    # Per voltage plots
    voltages = next(iter(qs.values())).grid["voltage"]
    voltages2 = (
        next(iter(qs.values())).grid["voltage2"]
        if params.use_voltage2
        else torch.zeros((1,))
    )

    # Separate voltage: Not in use
    # for voltage_index, voltage in enumerate(voltages):
    #     if (
    #         params.plot_each_voltage <= 0
    #         or voltage_index % params.plot_each_voltage != 0
    #     ):
    #         continue

    #     voltage_path_prefix = f"{path_prefix}{voltage:.2f}V/"
    #     os.makedirs(voltage_path_prefix, exist_ok=True)
    #     voltage_index_dict = {"voltage": [voltage_index]}

    # Separate voltage2: Not in use

    # Separate all combinations of voltages
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

        for voltage2_index, voltage2 in enumerate(voltages2):
            if voltage2_index % params.plot_each_voltage != 0:
                continue

            voltage_path_prefix = f"{path_prefix}{voltage:.2f}V/{voltage2:.2f}V2/"
            os.makedirs(voltage_path_prefix, exist_ok=True)
            voltage_index_dict = {"voltage": [voltage_index]}
            if params.use_voltage2:
                voltage_index_dict["voltage2"] = [voltage2_index]

            # TODO: some quantities in the following loop are not contact dependent,
            # take them out
            for contact in device.contacts:
                q_full = qs["bulk"]
                full_grid = Subgrid(q_full.grid, voltage_index_dict, copy_all=False)
                q_full = restrict_quantities(q_full, full_grid)
                full_grid_reduced = Subgrid(
                    full_grid, energies_index_dict, copy_all=False
                )
                q_full_reduced = restrict_quantities(q_full, full_grid_reduced)

                q_in = qs[contact.grid_name]
                in_boundary_grid = Subgrid(
                    q_in.grid, voltage_index_dict, copy_all=False
                )
                q_in = restrict_quantities(q_in, in_boundary_grid)
                in_boundary_grid_reduced = Subgrid(
                    in_boundary_grid, energies_index_dict, copy_all=False
                )
                q_in_reduced = restrict_quantities(q_in, in_boundary_grid_reduced)

                q_out = qs[f"boundary{contact.out_boundary_index}"]
                out_boundary_grid = Subgrid(
                    q_out.grid, voltage_index_dict, copy_all=False
                )
                q_out = restrict_quantities(q_out, out_boundary_grid)
                out_boundary_grid_reduced = Subgrid(
                    out_boundary_grid, energies_index_dict, copy_all=False
                )
                q_out_reduced = restrict_quantities(q_out, out_boundary_grid_reduced)

                phi_factor_reduced = 1  # / q_full_reduced[f"incoming_coeff_{contact}"]

                save_lineplot(
                    q_full["V_int"] + q_full["V_el"],
                    q_full.grid,
                    "V",
                    "x",
                    path_prefix=voltage_path_prefix,
                    quantity_unit=consts.EV,
                    quantity_unit_name="eV",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                )

                save_lineplot(
                    q_full["V_el"],
                    q_full.grid,
                    "V_el_old",
                    "x",
                    path_prefix=voltage_path_prefix,
                    quantity_unit=consts.EV,
                    quantity_unit_name="eV",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                )

                save_lineplot(
                    q_full["V_el_new"],
                    q_full.grid,
                    "V_el_new",
                    "x",
                    path_prefix=voltage_path_prefix,
                    quantity_unit=consts.EV,
                    quantity_unit_name="eV",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                )

                save_lineplot(
                    q_full["n"],
                    q_full.grid,
                    "n",
                    "x",
                    path_prefix=voltage_path_prefix,
                    quantity_unit=1 / consts.NM,
                    quantity_unit_name="1/nm",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                )

                save_lineplot(
                    q_full[f"fermi_integral_{contact}"],
                    q_full.grid,
                    f"fermi_integral_{contact}",
                    "DeltaE",
                    path_prefix=voltage_path_prefix,
                    quantity_unit=1 / consts.NM**2,
                    quantity_unit_name="nm^-2",
                    x_unit=consts.EV,
                    x_unit_name="eV",
                )

                # Wave function
                save_lineplot(
                    complex_abs2(q_full_reduced[f"phi_{contact}"] * phi_factor_reduced),
                    q_full_reduced.grid,
                    f"|phi_{contact}|^2",
                    "x",
                    "DeltaE",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    lines_unit=consts.EV,
                    lines_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )
                save_lineplot(
                    torch.real(q_full_reduced[f"phi_{contact}"] * phi_factor_reduced),
                    q_full_reduced.grid,
                    f"Re(phi_{contact})",
                    "x",
                    "DeltaE",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    lines_unit=consts.EV,
                    lines_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )
                save_lineplot(
                    torch.imag(q_full_reduced[f"phi_{contact}"] * phi_factor_reduced),
                    q_full_reduced.grid,
                    f"Im(phi_{contact})",
                    "x",
                    "DeltaE",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    lines_unit=consts.EV,
                    lines_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )
                visualization.save_complex_polar_plot(
                    q_full_reduced[f"phi_{contact}"] * phi_factor_reduced,
                    q_full_reduced.grid,
                    f"phi_{contact}",
                    "x",
                    "DeltaE",
                    lines_unit=consts.EV,
                    lines_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )
                save_heatmap(
                    q_full[f"DOS_{contact}"],
                    q_full.grid,
                    f"DOS_{contact}",
                    "x",
                    "DeltaE",
                    quantity_unit=1 / consts.NM / consts.EV,
                    quantity_unit_name="1/nm/eV",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    y_unit=consts.EV,
                    y_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )
                save_lineplot(
                    q_full_reduced[f"DOS_{contact}"],
                    q_full_reduced.grid,
                    f"DOS_{contact}",
                    "x",
                    "DeltaE",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    lines_unit=consts.EV,
                    lines_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )

                # Transmission and reflection probabilities
                save_lineplot(
                    complex_abs2(
                        q_full[f"transmitted_coeff_{contact}"]
                        / q_full[f"incoming_coeff_{contact}"]
                    ),
                    q_full.grid,
                    f"transmitted_coeff_{contact}",
                    "DeltaE",
                    x_quantity=q_full[f"E_{contact}"],
                    x_label="E",
                    x_unit=consts.EV,
                    x_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )

                fig, ax = plt.subplots()
                add_lineplot(
                    ax,
                    q_full[f"R_{contact}"],
                    q_full.grid,
                    "Reflection probability",
                    "DeltaE",
                    x_quantity=q_full[f"E_{contact}"],
                    x_unit=consts.EV,
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
                    x_unit=consts.EV,
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
                    quantity_unit=consts.COULOMB
                    / consts.SECOND
                    / consts.CM**2
                    / consts.EV,
                    quantity_unit_name="A/cm^2/eV",
                    x_unit=consts.EV,
                    x_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )
                save_heatmap(
                    q_full[f"I_spectrum_xdep_{contact}"],
                    q_full.grid,
                    f"I_spectrum_xdep_{contact}",
                    "x",
                    "DeltaE",
                    quantity_unit=consts.COULOMB
                    / consts.SECOND
                    / consts.CM**2
                    / consts.EV,
                    quantity_unit_name="A/cm^2/eV",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    y_unit=consts.EV,
                    y_unit_name="eV",
                    path_prefix=voltage_path_prefix,
                )

                save_lineplot(
                    q_full["n"],
                    q_full.grid,
                    "n",
                    "x",
                    quantity_unit=1 / consts.CM**3,
                    quantity_unit_name="1/cm$^3$",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    path_prefix=voltage_path_prefix,
                )

                save_lineplot(
                    q_full["doping"] - q_full["n"],
                    q_full.grid,
                    "Total Charge Density",
                    "x",
                    quantity_unit=1 / consts.CM**3,
                    quantity_unit_name="q/cm$^3$",
                    x_unit=consts.NM,
                    x_unit_name="nm",
                    path_prefix=voltage_path_prefix,
                )

                # save_lineplot(
                #     q_full["V_el"],
                #     q_full.grid,
                #     "V_el",
                #     "x",
                #     quantity_unit=consts.EV,
                #     quantity_unit_name="eV",
                #     x_unit=consts.NM,
                #     x_unit_name="nm",
                #     path_prefix=voltage_path_prefix,
                # )

                if not params.extra_plots:
                    continue

                extra_prefix = voltage_path_prefix + "extra/"
                os.makedirs(extra_prefix, exist_ok=True)

                for i in range(1, device.n_layers + 1):
                    q_layer = qs[f"bulk{i}"]
                    grid_layer = Subgrid(
                        q_layer.grid, voltage_index_dict, copy_all=False
                    )
                    q_layer = restrict_quantities(q_layer, grid_layer)
                    grid_layer_reduced = Subgrid(
                        grid_layer, energies_index_dict, copy_all=False
                    )
                    q_layer_reduced = restrict_quantities(q_layer, grid_layer_reduced)

                    cs = ("a", "b") if params.use_phi_one else ("a",)
                    for c in cs:
                        save_lineplot(
                            torch.real(q_layer_reduced[f"{c}_phase{i}_{contact}"]),
                            q_layer_reduced.grid,
                            f"Re[{c}_phase{i}_{contact}]",
                            "x",
                            "DeltaE",
                            x_unit=consts.NM,
                            x_unit_name="nm",
                            path_prefix=extra_prefix,
                        )
                        save_lineplot(
                            torch.imag(q_layer_reduced[f"{c}_phase{i}_{contact}"]),
                            q_layer_reduced.grid,
                            f"Im[{c}_phase{i}_{contact}]",
                            "x",
                            "DeltaE",
                            x_unit=consts.NM,
                            x_unit_name="nm",
                            path_prefix=extra_prefix,
                        )
                        save_lineplot(
                            complex_abs2(q_layer_reduced[f"{c}_phase{i}_{contact}"]),
                            q_layer_reduced.grid,
                            f"|{c}_phase{i}_{contact}|^2",
                            "x",
                            "DeltaE",
                            x_unit=consts.NM,
                            x_unit_name="nm",
                            path_prefix=extra_prefix,
                        )

                        save_lineplot(
                            torch.real(q_layer_reduced[f"{c}_output{i}_{contact}"]),
                            q_layer_reduced.grid,
                            f"Re[{c}_output{i}_{contact}]",
                            "x",
                            "DeltaE",
                            x_unit=consts.NM,
                            x_unit_name="nm",
                            path_prefix=extra_prefix,
                        )
                        save_lineplot(
                            torch.imag(q_layer_reduced[f"{c}_output{i}_{contact}"]),
                            q_layer_reduced.grid,
                            f"Im[{c}_output{i}_{contact}]",
                            "x",
                            "DeltaE",
                            x_unit=consts.NM,
                            x_unit_name="nm",
                            path_prefix=extra_prefix,
                        )
                        save_heatmap(
                            torch.real(q_layer[f"{c}_output{i}_{contact}"]),
                            q_layer.grid,
                            f"Re[{c}_output{i}_{contact}]",
                            "x",
                            "DeltaE",
                            x_unit=consts.NM,
                            x_unit_name="nm",
                            y_unit=consts.EV,
                            y_unit_name="eV",
                            path_prefix=extra_prefix,
                        )
                        save_heatmap(
                            torch.imag(q_layer[f"{c}_output{i}_{contact}"]),
                            q_layer.grid,
                            f"Im[{c}_output{i}_{contact}]",
                            "x",
                            "DeltaE",
                            x_unit=consts.NM,
                            x_unit_name="nm",
                            y_unit=consts.EV,
                            y_unit_name="eV",
                            path_prefix=extra_prefix,
                        )

                    save_lineplot(
                        torch.real(q_layer_reduced[f"phi_zero{i}_{contact}"]),
                        q_layer_reduced.grid,
                        f"Re[phi_zero{i}_{contact}]",
                        "x",
                        "DeltaE",
                        x_unit=consts.NM,
                        x_unit_name="nm",
                        path_prefix=extra_prefix,
                    )
                    save_lineplot(
                        torch.imag(q_layer_reduced[f"phi_zero{i}_{contact}"]),
                        q_layer_reduced.grid,
                        f"Im[phi_zero{i}_{contact}]",
                        "x",
                        "DeltaE",
                        x_unit=consts.NM,
                        x_unit_name="nm",
                        path_prefix=extra_prefix,
                    )
                    save_lineplot(
                        complex_abs2(q_layer_reduced[f"phi_zero{i}_{contact}"]),
                        q_layer_reduced.grid,
                        f"|phi_zero{i}_{contact}|^2",
                        "x",
                        "DeltaE",
                        x_unit=consts.NM,
                        x_unit_name="nm",
                        path_prefix=extra_prefix,
                    )
