# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Tuple, Callable

from kolpinn import model
from kolpinn.model import MultiModel

from classes import Device
import physical_constants as consts
import parameters as params


def get_trained_models(
    device: Device,
    *,
    dx_dict: Dict[str, float],
) -> Tuple[Sequence[MultiModel], Sequence[str]]:
    """
    Return models that represent the neural networks.

    Models:
        a/b_output{i}_{contact}: The NN output.
    """

    trained_models: list[MultiModel] = []
    trained_models_labels: list[str] = []

    N = device.n_layers

    # Layers
    for i in range(1, N + 1):
        layer_grids = (
            [f"boundary{i - 1}" + dx_string for dx_string in dx_dict.keys()]
            + [f"bulk{i}"]
            + [f"boundary{i}" + dx_string for dx_string in dx_dict.keys()]
        )
        x_left = device.boundaries[i - 1]
        x_right = device.boundaries[i]

        inputs_labels = []
        if params.continuous_voltage:
            inputs_labels.append("voltage")
        if params.continuous_energy:
            inputs_labels.append(
                "DeltaE"
            )  # TODO: check whether E_L/E_R or DeltaE should be provided (required_quantities_labels would need to be changed as well)
        inputs_labels.append("x")

        model_transformations: Dict[str, Callable] = {
            "x": lambda x, q, x_left=x_left, x_right=x_right: (x - x_left)
            / (x_right - x_left),
            "DeltaE": lambda E, q: E / consts.EV,
        }

        for contact in device.contacts:
            cs = ("a", "b") if params.use_phi_one else ("a",)
            for c in cs:
                nn_model = model.SimpleNNModel(
                    inputs_labels,
                    params.activation_function,
                    n_neurons_per_hidden_layer=params.n_neurons_per_hidden_layer,
                    n_hidden_layers=params.n_hidden_layers,
                    model_dtype=params.model_dtype,
                    output_dtype=params.si_complex_dtype,
                    device=params.device,
                )

                output_transformation = None
                # DEBUG: Free space solution w/ correct tensor shape
                # if (contact.name == "L" and c == "a") or (
                #     contact.name == "R" and c == "b"
                # ):
                #     output_transformation = lambda quantity, q: quantity * 0 + 1
                # else:
                #     output_transformation = lambda quantity, q: quantity * 0

                c_model = model.TransformedModel(
                    nn_model,
                    input_transformations=model_transformations,
                    output_transformation=output_transformation,
                )
                trained_models.append(
                    model.get_combined_multi_model(
                        c_model,
                        f"{c}_output{i}_{contact}",
                        layer_grids,
                        combined_dimension_name="x",
                        required_quantities_labels=["voltage", "DeltaE", "x"],
                    )
                )
                trained_models_labels.append(f"{c}_output{i}_{contact}")

    return trained_models, trained_models_labels
