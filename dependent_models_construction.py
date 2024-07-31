# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Tuple
import torch

from kolpinn import model
from kolpinn.model import MultiModel

import parameters as params
import physics
from classes import Device
import loss
import transformations as trafos


def get_dependent_models(
    device: Device,
    *,
    dx_dict: Dict[str, float],
) -> Tuple[Sequence[MultiModel], Dict[str, Sequence[str]]]:
    """
    Get the weight-dependent parameters and a dict containing the names
    of the quantities used as losses
    """

    dependent_models: list[MultiModel] = []
    loss_quantities: Dict[str, list[str]] = {}

    N = device.n_layers
    layer_indices_dict = {"L": range(N + 1, -1, -1), "R": range(0, N + 2)}

    # Add the coeffs to `dependent_models` layer by layer
    for contact in device.contacts:
        for i in layer_indices_dict[contact.name]:
            bulk = f"bulk{i}"
            boundary_in = f"boundary{contact.get_in_boundary_index(i)}"
            boundary_out = f"boundary{contact.get_out_boundary_index(i)}"
            is_in_contact = i == contact.index
            is_out_contact = i == contact.out_index
            is_contact = is_in_contact or is_out_contact
            bulks = [] if is_contact else [bulk]
            boundaries_in = (
                []
                if is_in_contact
                else [boundary_in + dx_string for dx_string in dx_dict.keys()]
            )
            boundaries_out = (
                []
                if is_out_contact
                else [boundary_out + dx_string for dx_string in dx_dict.keys()]
            )
            single_boundaries = []
            if not is_in_contact:
                single_boundaries.append(boundary_in)
            if not is_out_contact:
                single_boundaries.append(boundary_out)

            if not is_contact:
                for c in ("a", "b"):
                    dependent_models.append(
                        trafos.get_dx_model(
                            "multigrid" if params.fd_first_derivatives else "exact",
                            f"{c}_output{i}_{contact}",
                            boundary_out,
                        )
                    )

            if not is_out_contact:
                dependent_models.append(
                    MultiModel(
                        lambda qs, i=i, contact=contact: trafos.factors_trafo(
                            qs, i, contact
                        ),
                        f"factors{i}_{contact}",
                    )
                )

                for grid_name in boundaries_in + bulks + boundaries_out:
                    dependent_models.append(
                        MultiModel(
                            lambda qs, contact=contact, grid_name=grid_name, i=i: trafos.add_coeffs(
                                qs, contact, grid_name, i
                            ),
                            f"coeffs{i}",
                        )
                    )

            if not is_contact:
                for c in ("a", "b"):
                    for grid_name in boundaries_in:
                        dependent_models.append(
                            model.get_multi_model(
                                model.FunctionModel(
                                    lambda q, c=c, i=i, contact=contact: q[
                                        f"{c}{i}_{contact}"
                                    ]
                                    * q[f"{c}_propagation_factor{i}_{contact}"]
                                ),
                                f"{c}{i}_propagated_{contact}",
                                grid_name,
                            )
                        )

                    dependent_models.append(
                        trafos.get_dx_model(
                            "multigrid" if params.fd_first_derivatives else "exact",
                            f"{c}{i}_propagated_{contact}",
                            boundary_in,
                        )
                    )

    # Derived quantities
    # Input contact (includes global quantities)
    for contact in device.contacts:
        dependent_models.append(
            MultiModel(
                trafos.TR_trafo,
                f"T/R_{contact}",
                kwargs={"contact": contact},
            )
        )

        dependent_models.append(
            MultiModel(
                trafos.I_contact_trafo,
                f"I_{contact}",
                kwargs={"contact": contact},
            )
        )

    dependent_models.append(
        MultiModel(
            trafos.I_trafo,
            "I",
            kwargs={"contacts": device.contacts},
        )
    )

    # Bulk
    for i in range(1, N + 1):
        bulk_name = f"bulk{i}"
        loss_quantities[bulk_name] = []

        for contact in device.contacts:
            dependent_models.append(
                model.get_multi_model(
                    model.FunctionModel(
                        lambda q, i=i, contact=contact: (
                            q[f"a{i}_{contact}"] * q[f"a_phase{i}_{contact}"]
                            + q[f"b{i}_{contact}"] * q[f"b_phase{i}_{contact}"]
                        )
                    ),
                    f"phi{i}_{contact}",
                    bulk_name,
                )
            )

            dependent_models.append(
                trafos.get_dx_model(
                    "singlegrid" if params.fd_first_derivatives else "exact",
                    f"phi{i}_{contact}",
                    bulk_name,
                )
            )

            dependent_models.append(
                model.get_multi_model(
                    model.FunctionModel(
                        lambda q, i=i, contact=contact: torch.imag(
                            physics.H_BAR
                            * torch.conj(q[f"phi{i}_{contact}"])
                            * q[f"phi{i}_{contact}_dx"]
                            / q[f"m_eff{i}"]
                        ),
                    ),
                    f"j{i}_{contact}",
                    bulk_name,
                )
            )

            dependent_models.append(
                model.MultiModel(
                    loss.j_loss_trafo,
                    f"j_loss{i}_{contact}",
                    kwargs={"i": i, "N": N, "contact": contact},
                )
            )
            loss_quantities[bulk_name].append(f"j_loss{i}_{contact}")

    for contact in device.contacts:
        dependent_models.append(
            MultiModel(
                trafos.to_full_trafo,
                f"phi_{contact}",
                kwargs={
                    "N": N,
                    "label_fn": lambda i, *, contact=contact: f"phi{i}_{contact}",
                    "quantity_label": f"phi_{contact}",
                },
            )
        )
        dependent_models.append(
            MultiModel(
                trafos.dos_trafo,
                f"DOS_{contact}",
                kwargs={"contact": contact},
            )
        )
        dependent_models.append(
            MultiModel(
                trafos.n_contact_trafo,
                f"n_{contact}",
                kwargs={"contact": contact},
            )
        )

    dependent_models.append(
        MultiModel(
            trafos.n_trafo,
            "n",
            kwargs={"contacts": device.contacts},
        )
    )
    dependent_models.append(
        MultiModel(
            trafos.V_electrostatic_trafo,
            "V_el",
        )
    )
    dependent_models.append(
        MultiModel(
            trafos.to_bulks_trafo,
            "V_el_distribution",
            kwargs={
                "N": N,
                "label_fn": lambda i, *, contact=contact: f"V_el{i}",
                "quantity_label": "V_el",
            },
        )
    )

    # Bulk again
    for i in range(1, N + 1):
        bulk_name = f"bulk{i}"
        for contact in device.contacts:
            dependent_models.append(
                MultiModel(
                    loss.SE_loss_trafo,
                    f"SE_loss{i}_{contact}",
                    kwargs={
                        "qs_full": None,
                        "with_grad": True,
                        "i": i,
                        "N": N,
                        "contact": contact,
                    },
                )
            )
            loss_quantities[bulk_name].append(f"SE_loss{i}_{contact}")

    # TODO: type "widening"
    # assert type(dependent_models) is Sequence[MultiModel], type(dependent_models)
    # assert type(loss_quantities) is Dict[str,Sequence[str]], type(used_losses)
    return dependent_models, loss_quantities
