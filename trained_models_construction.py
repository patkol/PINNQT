# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Tuple, Callable
import torch

from kolpinn import grids
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import model
from kolpinn.model import MultiModel

from classes import Device
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

    required_quantities_labels = ["voltage", "DeltaE", "x"]
    if params.use_voltage2:
        required_quantities_labels.append("voltage2")

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
            if params.use_voltage2:
                inputs_labels.append("voltage2")
        if params.continuous_energy:
            inputs_labels.append(
                "DeltaE"
            )  # TODO: check whether E_L/E_R or DeltaE should be provided (required_quantities_labels would need to be changed as well)
        inputs_labels.append("x")

        model_transformations: Dict[str, Callable] = {
            "x": lambda x, q, x_left=x_left, x_right=x_right: (
                x - (x_left + x_right) / 2
            )
            / params.x_input_scale,
            "DeltaE": lambda E, q: E / params.E_input_scale
            + (
                0
                if params.E_input_scale_sqrt is None
                else torch.sqrt(E / params.E_input_scale_sqrt)
            ),
            "voltage": lambda U, q: U / params.U_input_scale,
        }
        if params.use_voltage2:
            model_transformations["voltage2"] = model_transformations["voltage"]

        for contact in device.contacts:
            # The part below is partly a code duplication with `SimpleNNModel`,
            # `TransformedModel`, and
            # `get_combined_multi_model` from kolpinn.model.
            # It is reimplemented here to allow for two outputs (a and b)
            nn = model.SimpleNetwork(
                params.activation_function,
                n_inputs=len(inputs_labels),
                n_outputs=4 if params.use_phi_one else 2,
                n_neurons_per_hidden_layer=params.n_neurons_per_hidden_layer,
                n_hidden_layers=params.n_hidden_layers,
                dtype=params.model_dtype,
            )

            def qs_trafo(
                qs: Dict[str, QuantityDict],
                *,
                grid_names=layer_grids,
                combined_dimension_name="x",
                required_quantities_labels=required_quantities_labels,
                n_inputs=len(inputs_labels),
                nn=nn,
                model_transformations=model_transformations,
                i=i,
                contact=contact,
            ):
                child_grids = dict(
                    (grid_name, qs[grid_name].grid) for grid_name in grid_names
                )
                supergrid = grids.Supergrid(
                    child_grids,
                    combined_dimension_name,
                    copy_all=False,
                )
                q = QuantityDict(supergrid)
                for label in required_quantities_labels:
                    q[label] = quantities.combine_quantity(
                        [qs[child_name][label] for child_name in grid_names],
                        list(supergrid.subgrids.values()),
                        supergrid,
                    )

                # Evaluate the NN

                # inputs_tensor[gridpoint, input quantity]
                inputs_tensor = torch.zeros(
                    (q.grid.n_points, n_inputs),
                    dtype=params.model_dtype,
                )
                for input_index, label in enumerate(inputs_labels):
                    transformed_input = model_transformations[label](q[label], q)
                    inputs_tensor[:, input_index] = quantities.expand_all_dims(
                        transformed_input,
                        q.grid,
                    ).flatten()
                output = nn(inputs_tensor)
                a_output = torch.view_as_complex(output[..., :2])
                a_output = a_output.reshape(q.grid.shape)
                a_output = a_output.to(params.si_complex_dtype)
                if params.use_phi_one:
                    b_output = torch.view_as_complex(output[..., 2:])
                    b_output = b_output.reshape(q.grid.shape)
                    b_output = b_output.to(params.si_complex_dtype)
                # DEBUG: Output 1
                # a_output = 0 * a_output + 1
                # b_output = 0 * b_output + 1

                for grid_name in grid_names:
                    qs[grid_name][f"a_output{i}_{contact}"] = quantities.restrict(
                        a_output,
                        supergrid.subgrids[grid_name],
                    )
                    if not params.use_phi_one:
                        continue
                    qs[grid_name][f"b_output{i}_{contact}"] = quantities.restrict(
                        b_output,
                        supergrid.subgrids[grid_name],
                    )

                return qs

            nn_model = MultiModel(
                qs_trafo,
                f"NN{i}_{contact}",
                parameters_in=list(nn.parameters()),
                networks_in=[nn],
            )
            trained_models.append(nn_model)
            trained_models_labels.append(f"NN{i}_{contact}")

    return trained_models, trained_models_labels
