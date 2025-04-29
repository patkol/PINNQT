# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Tuple
import torch

from kolpinn import model
from kolpinn.model import MultiModel

from classes import Device
import physical_constants as consts
import parameters as params
import formulas
import loss
import transformations as trafos


def get_loss_models(
    device: Device,
    *,
    dx_dict: Dict[str, float],
) -> Tuple[Sequence[MultiModel], Dict[str, Sequence[str]]]:
    """
    Get the weight-dependent models needed to calculate the loss
    and a dict containing the names of the quantities used as losses

    Models:
        phi_zero/one{i}_{contact}: Ansaetze modulated by the NN outputs
        if params.hard_bc_dir != 0:
            phi_zero/one{i}_{contact}_dx: Their derivative, only computed at the
                input/output boundary of each layer if params.hard_bc_dir == +/-1
        phi{i}_{contact}[_dx]: Wave function and its derivative
        incoming/reflected/transmitted_coeff_{contact}: Prefactors to the corresponding
            contact plane waves (if not fixed already in the constant models)
        j{i}_{contact}: current density
        j/SE/cc/wc_loss_{i}_{contact}: losses
    """

    loss_models: list[MultiModel] = []
    loss_quantities: Dict[str, list[str]] = {}

    N = device.n_layers

    for i in range(1, N + 1):
        loss_quantities[f"bulk{i}"] = []
    if params.soft_bc or params.soft_bc_output:
        # Only the contact boundaries have losses if only soft_bc_output is True
        boundary_indices = range(0, N + 1) if params.soft_bc else (0, N)
        for i in boundary_indices:
            loss_quantities[f"boundary{i}"] = []

    # Propagate from the input to the output / output to the input, layer by layer
    for contact in device.contacts:
        in_layer_index = contact.get_out_layer_index(contact.in_boundary_index)
        out_layer_index = contact.get_in_layer_index(contact.out_boundary_index)
        layer_indices_range = None
        if params.hard_bc_dir == -1:
            layer_indices_range = range(
                out_layer_index, contact.index, -contact.direction
            )
        else:
            layer_indices_range = range(
                in_layer_index, contact.out_index, contact.direction
            )

        for i in layer_indices_range:
            bulk = f"bulk{i}"
            bulks = [bulk]
            boundary_in = f"boundary{contact.get_in_boundary_index(i)}"
            boundary_out = f"boundary{contact.get_out_boundary_index(i)}"
            boundaries_in = [boundary_in + dx_string for dx_string in dx_dict.keys()]
            boundaries_out = [boundary_out + dx_string for dx_string in dx_dict.keys()]

            loss_models.append(
                MultiModel(
                    trafos.phi_zero_one_trafo,
                    f"phi_zero/one{i}_{contact}",
                    kwargs={
                        "i": i,
                        "contact": contact,
                        "grid_names": boundaries_in + bulks + boundaries_out,
                    },
                )
            )

            # OPTIM: only evaluate where necessary
            if params.hard_bc_dir != 0:
                bc_boundary = boundary_in if params.hard_bc_dir == 1 else boundary_out
                loss_models.append(
                    formulas.get_dx_model(
                        "multigrid" if params.fd_first_derivatives else "exact",
                        f"phi_zero{i}_{contact}",
                        bc_boundary,
                    )
                )
                if params.use_phi_one:
                    loss_models.append(
                        formulas.get_dx_model(
                            "multigrid" if params.fd_first_derivatives else "exact",
                            f"phi_one{i}_{contact}",
                            bc_boundary,
                        )
                    )

            # OPTIM: only evaluate where necessary (don't need to take derivatives
            # of phi on one side with hard BC)
            loss_models.append(
                MultiModel(
                    trafos.phi_trafo,
                    f"phi{i}_{contact}",
                    kwargs={
                        "i": i,
                        "contact": contact,
                        "grid_names": boundaries_in + bulks + boundaries_out,
                    },
                )
            )

            if params.learn_phi_prime:
                # The derivative has been computed in phi_trafo already
                continue

            # phi_dx in bulk for current density
            loss_models.append(
                formulas.get_dx_model(
                    "singlegrid" if params.fd_first_derivatives else "exact",
                    f"phi{i}_{contact}",
                    bulk,
                )
            )
            # phi_dx on boundary for BC
            # OPTIM: only calculate the needed BC based on which BC are applied
            for boundary in (boundary_in, boundary_out):
                loss_models.append(
                    formulas.get_dx_model(
                        "multigrid" if params.fd_first_derivatives else "exact",
                        f"phi{i}_{contact}",
                        boundary,
                    )
                )

        loss_models.append(
            MultiModel(
                trafos.contact_coeffs_trafo,
                f"contact_coeffs_{contact}",
                kwargs={"contact": contact},
            )
        )

        # Bulk
        for i in range(1, N + 1):
            bulk_name = f"bulk{i}"

            loss_models.append(
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

            loss_models.append(
                model.MultiModel(
                    loss.j_loss_trafo,
                    f"j_loss{i}_{contact}",
                    kwargs={"i": i, "N": N, "contact": contact},
                )
            )
            loss_quantities[bulk_name].append(f"j_loss{i}_{contact}")

            loss_models.append(
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

        if not params.soft_bc and not params.soft_bc_output:
            continue

        boundary_indices = (
            range(0, N + 1) if params.soft_bc else (contact.out_boundary_index,)
        )

        # Boundaries
        for i in boundary_indices:
            boundary_name = f"boundary{i}"
            loss_models.append(
                MultiModel(
                    loss.cc_loss_trafo,
                    f"cc_loss{i}_{contact}",
                    kwargs={
                        "i": i,
                        "contact": contact,
                    },
                )
            )
            loss_quantities[boundary_name].append(f"cc_loss{i}_{contact}")

        if not params.soft_bc:
            continue

        # Inner boundaries
        for i in range(1, N):
            boundary_name = f"boundary{i}"
            loss_models.append(
                MultiModel(
                    loss.wc_loss_trafo,
                    f"wc_loss{i}_{contact}",
                    kwargs={
                        "i": i,
                        "contact": contact,
                    },
                )
            )
            loss_quantities[boundary_name].append(f"wc_loss{i}_{contact}")

    # TODO: type "widening"
    # assert type(loss_models) is Sequence[MultiModel], type(loss_models)
    # assert type(loss_quantities) is Dict[str,Sequence[str]], type(used_losses)
    return loss_models, loss_quantities
