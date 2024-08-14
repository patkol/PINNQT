# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict
import numpy as np
import torch

from kolpinn import model
from kolpinn.model import Model, MultiModel

import physical_constants as consts
import parameters as params
import physics
from classes import Device
import transformations as trafos


def get_constant_models(
    device: Device,
    *,
    dx_dict: Dict[str, float],
) -> Sequence[MultiModel]:
    const_models: list[MultiModel] = []

    N = device.n_layers

    layer_indep_const_models_dict = {}
    layer_indep_const_models_dict["E_L"] = model.FunctionModel(
        lambda q: q["DeltaE"],
    )
    layer_indep_const_models_dict["E_R"] = model.FunctionModel(
        lambda q: q["DeltaE"] - consts.Q_E * q["voltage"],
    )

    # const_models_dict[i][name] = model
    const_models_dict: Dict[int, Dict[str, model.Model]] = dict(
        (i, {}) for i in range(0, N + 2)
    )

    # Layers and contacts
    for i, models_dict in const_models_dict.items():
        models_dict[f"V_int{i}"] = model.get_model(
            device.potentials[i],
            model_dtype=params.si_real_dtype,
            output_dtype=params.si_real_dtype,
        )
        models_dict[f"V_el_approx{i}"] = model.FunctionModel(
            lambda q, i=i: trafos.get_V_voltage(
                q, device.device_start, device.device_end
            ),
        )
        models_dict[f"m_eff{i}"] = model.get_model(
            device.m_effs[i],
            model_dtype=params.si_real_dtype,
            output_dtype=params.si_real_dtype,
        )
        models_dict[f"doping{i}"] = model.get_model(
            device.dopings[i],
            model_dtype=params.si_real_dtype,
            output_dtype=params.si_real_dtype,
        )
        models_dict[f"permittivity{i}"] = model.get_model(
            device.permittivities[i],
            model_dtype=params.si_real_dtype,
            output_dtype=params.si_real_dtype,
        )

        for contact in device.contacts:
            x_in = device.boundaries[contact.get_in_boundary_index(i)]
            x_out = device.boundaries[contact.get_out_boundary_index(i)]

            models_dict[f"k{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, contact=contact: trafos.k_function(q, i, contact),
            )
            models_dict[f"smooth_k{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, contact=contact: q[f"k{i}_{contact}"]
                if i in (0, N + 1)
                else trafos.smooth_k_function(q, i, contact),
            )
            # The shifts by x_out are important for
            # energies smaller than V, it keeps them from exploding.
            models_dict[f"a_phase{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, x_out=x_out, contact=contact: torch.exp(
                    1j * q[f"smooth_k{i}_{contact}"] * (q["x"] - x_out)
                ),
            )
            # b_phase explodes for large layers and imaginary smooth_k
            models_dict[f"b_phase{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, x_out=x_out, contact=contact: torch.exp(
                    -1j * q[f"smooth_k{i}_{contact}"] * (q["x"] - x_out)
                ),
            )
            models_dict[f"a_propagation_factor{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, x_in=x_in, x_out=x_out, contact=contact: torch.exp(
                    1j * q[f"smooth_k{i}_{contact}"] * (x_in - x_out)
                ),
            )
            models_dict[f"b_propagation_factor{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, x_in=x_in, x_out=x_out, contact=contact: torch.exp(
                    -1j * q[f"smooth_k{i}_{contact}"] * (x_in - x_out)
                ),
            )

    zero_model = model.ConstModel(0, model_dtype=params.si_real_dtype)
    one_model = model.ConstModel(1, model_dtype=params.si_real_dtype)

    # Both contacts
    const_models_dict[0]["V_el0"] = model.ConstModel(
        0,
        model_dtype=params.si_real_dtype,
    )
    const_models_dict[N + 1][f"V_el{N+1}"] = model.FunctionModel(
        lambda q: -q["voltage"] * consts.EV,
    )

    for contact in device.contacts:
        for i in (contact.index, contact.out_index):
            for c in ("a", "b"):
                const_models_dict[i][f"{c}_output{i}_{contact}"] = one_model
                const_models_dict[i][f"{c}_output{i}_{contact}_dx"] = zero_model
            # V_el_approx is exact at the contacts
            const_models_dict[i][f"v{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, contact=contact: torch.sqrt(
                    (
                        2
                        * (q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el{i}"])
                        / q[f"m_eff{i}"]
                    ).to(params.si_complex_dtype)
                )
            )

    # Output contact: Initial conditions
    const_models_dict[N + 1][f"a{N + 1}_L"] = one_model
    const_models_dict[N + 1][f"b{N + 1}_L"] = zero_model
    const_models_dict[0]["a0_R"] = zero_model
    const_models_dict[0]["b0_R"] = one_model
    for contact in device.contacts:
        i = contact.out_index
        for c in ("a", "b"):
            const_models_dict[i][f"{c}{i}_{contact}_dx"] = zero_model
            const_models_dict[i][f"{c}{i}_propagated_{contact}"] = const_models_dict[i][
                f"{c}{i}_{contact}"
            ]
            const_models_dict[i][f"{c}{i}_propagated_{contact}_dx"] = const_models_dict[
                i
            ][f"{c}{i}_{contact}_dx"]

    # Input contact
    for contact in device.contacts:
        # OPTIM: move to 'bulk' such that it does not have to be computed
        #        for the pdx & mdx grids separately
        i = contact.index
        const_models_dict[i][f"dE_dk{i}_{contact}"] = model.FunctionModel(
            lambda q, i=i, contact=contact: torch.sqrt(
                2
                * consts.H_BAR**2
                * (q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el{i}"])
                / q[f"m_eff{i}"]
            ),
        )
        const_models_dict[i][f"E_fermi_{contact}"] = model.FunctionModel(
            lambda q, i=i: trafos.get_E_fermi(q, i=i),
        )
        const_models_dict[i][f"fermi_integral_{contact}"] = model.FunctionModel(
            lambda q, i=i, contact=contact: (
                q[f"m_eff{i}"]
                / (np.pi * consts.H_BAR**2 * physics.BETA)
                * torch.log(
                    1
                    + torch.exp(
                        physics.BETA * (q[f"E_fermi_{contact}"] - q[f"E_{contact}"])
                    )
                )
            ),
        )

    # Full device
    grid_name = "bulk"
    for model_name, model_ in layer_indep_const_models_dict.items():
        const_models.append(
            model.get_multi_model(model_, model_name, grid_name),
        )

    # Layers
    for i in range(1, N + 1):
        grid_name = f"bulk{i}"
        models_dict: Dict[str, Model] = dict(
            layer_indep_const_models_dict, **const_models_dict[i]
        )
        for model_name, model_ in models_dict.items():
            const_models.append(model.get_multi_model(model_, model_name, grid_name))

    # Boundaries
    for i in range(0, N + 1):
        for dx_string in dx_dict.keys():
            grid_name = f"boundary{i}" + dx_string
            for model_name, model_ in layer_indep_const_models_dict.items():
                const_models.append(
                    model.get_multi_model(model_, model_name, grid_name)
                )
            for j in (i, i + 1):
                models_dict = const_models_dict[j]
                for model_name, model_ in models_dict.items():
                    const_models.append(
                        model.get_multi_model(model_, model_name, grid_name)
                    )

    # Constant MultiModels
    for contact in device.contacts:
        # OPTIM: unused
        const_models.append(
            model.MultiModel(
                trafos.j_exact_trafo,
                f"j_exact_{contact}",
                kwargs={"contact": contact},
            )
        )

    const_models.append(
        model.MultiModel(
            trafos.to_full_trafo,
            "m_eff",
            kwargs={
                "N": N,
                "label_fn": lambda i, *, contact=contact: f"m_eff{i}",
                "quantity_label": "m_eff",
            },
        )
    )
    const_models.append(
        model.MultiModel(
            trafos.to_full_trafo,
            "V_int",
            kwargs={
                "N": N,
                "label_fn": lambda i, *, contact=contact: f"V_int{i}",
                "quantity_label": "V_int",
            },
        )
    )
    const_models.append(
        model.MultiModel(
            trafos.to_full_trafo,
            "doping",
            kwargs={
                "N": N,
                "label_fn": lambda i, *, contact=contact: f"doping{i}",
                "quantity_label": "doping",
            },
        )
    )
    const_models.append(
        model.MultiModel(
            trafos.to_full_trafo,
            "permittivity",
            kwargs={
                "N": N,
                "label_fn": lambda i, *, contact=contact: f"permittivity{i}",
                "quantity_label": "permittivity",
            },
        )
    )

    return const_models
