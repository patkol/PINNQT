# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Callable
import torch

from kolpinn.quantities import QuantityDict
from kolpinn import model
from kolpinn.model import Model, MultiModel

from classes import Device
import physical_constants as consts
import parameters as params
import formulas
import transformations as trafos


def get_constant_models(
    device: Device,
    *,
    dx_dict: Dict[str, float],
    V_el_function: Callable[[QuantityDict], torch.Tensor],  # Maps q to V_el
) -> Sequence[MultiModel]:
    """
    Return models that are independent of the trained parameters.

    Models:
        E_{contact}: The energy levels when propagating from `contact`.
        V_int{i} / m_eff{i} / doping{i} / permittivity{i} / k{i}_{contact}:
            Physical constants in the whole device
        v{i}_{contact}: Physical constant computed only in the contacts, based on k
        V_el{i}: Based on the provided V_el_function. At the contacts, V_el{0} = 0 and
            V_el{N+1} = applied bias are forced (TODO: change that)
        a/b_phase{i}_{contact}: The ansatz that will be modulated by the NN
        E_fermi_{contact} / fermi_integral_{contact}: Global constants in "bulk"

        if params.force_unity_coeff:
            if params.hard_bc_dir == 1:
                incoming_coeff_{contact} = 1

            if params.hard_bc_dir == -1:
                transmitted_coeff_{contact} = 1
                phi{i}_{contact} = 1 and
                phi{i}_{contact}_dx = contact.direction * i * k
                at the output contact.
    """

    N = device.n_layers
    zero_model = model.ConstModel(0, model_dtype=params.si_real_dtype)
    one_model = model.ConstModel(1, model_dtype=params.si_real_dtype)

    layer_indep_const_models_dict = {}
    layer_indep_const_models_dict["E_L"] = model.FunctionModel(
        lambda q: q["DeltaE"],
    )
    layer_indep_const_models_dict["E_R"] = model.FunctionModel(
        lambda q: q["DeltaE"] - consts.Q_E * q["voltage"],
    )

    # const_models_dict[i][name] = model
    const_models_dict: Dict[int, Dict[str, Model]] = dict(
        (i, {}) for i in range(0, N + 2)
    )

    # Layers and contacts
    for i, models_dict in const_models_dict.items():
        models_dict[f"V_int{i}"] = model.get_model(
            device.potentials[i],
            model_dtype=params.si_real_dtype,
            output_dtype=params.si_real_dtype,
        )
        # Using an approximation of V_el initially
        models_dict[f"V_el{i}"] = model.FunctionModel(V_el_function)
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
            models_dict[f"k{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, contact=contact: formulas.get_k(q, i, contact),
            )

            if params.ansatz == "none":
                models_dict[f"a_phase{i}_{contact}"] = one_model
            if not params.use_phi_one:
                continue
            if params.ansatz == "none":
                models_dict[f"b_phase{i}_{contact}"] = one_model

    # Both contacts
    const_models_dict[0]["V_el0"] = zero_model
    const_models_dict[N + 1][f"V_el{N+1}"] = model.FunctionModel(
        lambda q: -q["voltage"] * consts.EV,
    )
    for contact in device.contacts:
        for i in (contact.index, contact.out_index):
            const_models_dict[i][f"v{i}_{contact}"] = model.FunctionModel(
                lambda q, i=i, contact=contact: consts.H_BAR
                * q[f"k{i}_{contact}"]
                / q[f"m_eff{i}"]
            )

        if params.hard_bc_dir == -1 and params.force_unity_coeff:
            # Output contact: Boundary conditions
            i = contact.out_index
            const_models_dict[i][f"phi{i}_{contact}"] = one_model
            const_models_dict[i][f"phi{i}_{contact}_dx"] = model.FunctionModel(
                lambda q, i=i, contact=contact: contact.direction
                * 1j
                * q[f"k{i}_{contact}"]
            )

    const_models: list[MultiModel] = []

    # Full device
    for model_name, model_ in layer_indep_const_models_dict.items():
        const_models.append(
            model.get_multi_model(model_, model_name, "bulk"),
        )

    # Layers
    for i in range(1, N + 1):
        grid_name = f"bulk{i}"
        layer_models_dict: Dict[str, Model] = dict(
            layer_indep_const_models_dict, **const_models_dict[i]
        )
        for model_name, model_ in layer_models_dict.items():
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

    # Full device again
    for contact in device.contacts:
        if params.force_unity_coeff:
            fixed_coeff = "transmitted" if params.hard_bc_dir == -1 else "incoming"
            const_models.append(
                model.get_multi_model(
                    one_model, f"{fixed_coeff}_coeff_{contact}", "bulk"
                ),
            )
        const_models.append(
            model.MultiModel(
                trafos.E_fermi_trafo,
                f"E_fermi_{contact}",
                kwargs={"contact": contact},
            )
        )
        const_models.append(
            model.MultiModel(
                trafos.fermi_integral_trafo,
                f"fermi_integral_{contact}",
                kwargs={"contact": contact},
            )
        )

    # Promoting quantities to full grid
    for contact in device.contacts:
        const_models.append(
            model.MultiModel(
                trafos.to_full_trafo,
                f"k_{contact}",
                kwargs={
                    "N": N,
                    "label_fn": lambda i, *, contact=contact: f"k{i}_{contact}",
                    "quantity_label": f"k_{contact}",
                },
            )
        )

        if params.ansatz not in ("wkb", "half_wkb"):
            continue

        const_models.append(
            model.MultiModel(
                trafos.wkb_phase_trafo,
                "wkb_phase",
                kwargs={
                    "contact": contact,
                    "N": N,
                    "dx_dict": dx_dict,
                    "smoothing_method": params.wkb_smoothing_method,
                    "smoothing_kwargs": params.wkb_smoothing_kwargs,
                },
            )
        )

    const_models.append(
        model.MultiModel(
            trafos.to_full_trafo,
            "V_el",
            kwargs={
                "N": N,
                "label_fn": lambda i: f"V_el{i}",
                "quantity_label": "V_el",
            },
        )
    )
    const_models.append(
        model.MultiModel(
            trafos.to_full_trafo,
            "m_eff",
            kwargs={
                "N": N,
                "label_fn": lambda i: f"m_eff{i}",
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
                "label_fn": lambda i: f"V_int{i}",
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
                "label_fn": lambda i: f"doping{i}",
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
                "label_fn": lambda i: f"permittivity{i}",
                "quantity_label": "permittivity",
            },
        )
    )

    return const_models
