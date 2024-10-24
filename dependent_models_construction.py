# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Tuple
import torch

from kolpinn import model
from kolpinn.model import MultiModel

import physical_constants as consts
import parameters as params
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

    # Propagate from the output to the input, layer by layer
    layer_indices_dict = {"L": range(N, 0, -1), "R": range(1, N + 1)}
    for contact in device.contacts:
        for i in layer_indices_dict[contact.name]:
            bulk = f"bulk{i}"
            bulks = [bulk]
            boundary_in = f"boundary{contact.get_in_boundary_index(i)}"
            boundary_out = f"boundary{contact.get_out_boundary_index(i)}"
            boundaries_in = [boundary_in + dx_string for dx_string in dx_dict.keys()]
            boundaries_out = [boundary_out + dx_string for dx_string in dx_dict.keys()]

            for grid_name in boundaries_in + bulks + boundaries_out:
                dependent_models.append(
                    model.get_multi_model(
                        model.FunctionModel(
                            lambda q, *, i=i, contact=contact: trafos.get_phi_zero(
                                q, i=i, contact=contact
                            )
                        ),
                        f"phi_zero{i}_{contact}",
                        grid_name,
                    )
                )
                # For d0 phi0 + d1 phi1
                dependent_models.append(
                    model.get_multi_model(
                        model.FunctionModel(
                            lambda q, *, i=i, contact=contact: trafos.get_phi_one(
                                q, i=i, contact=contact
                            )
                        ),
                        f"phi_one{i}_{contact}",
                        grid_name,
                    )
                )
            dependent_models.append(
                trafos.get_dx_model(
                    "multigrid" if params.fd_first_derivatives else "exact",
                    f"phi_zero{i}_{contact}",
                    boundary_out,
                )
            )
            # For d0 phi0 + d1 phi1
            dependent_models.append(
                trafos.get_dx_model(
                    "multigrid" if params.fd_first_derivatives else "exact",
                    f"phi_one{i}_{contact}",
                    boundary_out,
                )
            )
            dependent_models.append(
                MultiModel(
                    trafos.phi_trafo,
                    f"phi{i}_{contact}",
                    kwargs={
                        "i": i,
                        "contact": contact,
                        "grid_names": boundaries_in + bulks,
                    },
                )
            )
            dependent_models.append(
                trafos.get_dx_model(
                    "multigrid" if params.fd_first_derivatives else "exact",
                    f"phi{i}_{contact}",
                    boundary_in,
                )
            )
            # phi_dx in bulk for current density
            dependent_models.append(
                trafos.get_dx_model(
                    "singlegrid" if params.fd_first_derivatives else "exact",
                    f"phi{i}_{contact}",
                    bulk,
                )
            )

    # Derived quantities
    # Input contact (includes global quantities)
    for contact in device.contacts:
        dependent_models.append(
            MultiModel(
                trafos.contact_coeffs_trafo,
                f"contact_coeffs_{contact}",
                kwargs={"contact": contact},
            )
        )
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
                        lambda q, i=i, contact=contact: torch.imag(
                            consts.H_BAR
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
            # Ignoring the j loss due to complex energies
            # loss_quantities[bulk_name].append(f"j_loss{i}_{contact}")

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
