# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from collections.abc import Sequence

from kolpinn.model import MultiModel

from classes import Device
import parameters as params
import transformations as trafos


def get_eval_models(device: Device) -> Sequence[MultiModel]:
    """
    Get parameters which are used for the evaluation of the results, but not necessary
    for calculating the losses (these are in loss_models).

    Models:
        T/R_{contact}: Transmitted/reflected amplitude
        I/n[_{contact}]: Current/density [from one contact only]
        phi_{contact}: Wavefunction on "bulk"
        DOS_{contact}: density of states from one contact only
    """

    eval_models: list[MultiModel] = []

    N = device.n_layers

    # Derived quantities
    # Input contact (includes global quantities)
    for contact in device.contacts:
        eval_models.append(
            MultiModel(
                trafos.TR_trafo,
                f"T/R_{contact}",
                kwargs={"contact": contact},
            )
        )
        eval_models.append(
            MultiModel(
                trafos.I_contact_trafo,
                f"I_{contact}",
                kwargs={"contact": contact},
            )
        )

    eval_models.append(
        MultiModel(
            trafos.I_trafo,
            "I",
            kwargs={"contacts": device.contacts},
        )
    )

    if not params.use_induced_V_el:
        for contact in device.contacts:
            eval_models.append(
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
            eval_models.append(
                MultiModel(
                    trafos.dos_trafo,
                    f"DOS_{contact}",
                    kwargs={"contact": contact},
                )
            )
            eval_models.append(
                MultiModel(
                    trafos.n_contact_trafo,
                    f"n_{contact}",
                    kwargs={"contact": contact},
                )
            )

        eval_models.append(
            MultiModel(
                trafos.n_trafo,
                "n",
                kwargs={"contacts": device.contacts},
            )
        )
        eval_models.append(
            MultiModel(
                trafos.V_electrostatic_trafo,
                "V_el_new",
                kwargs={"contacts": device.contacts, "N": N},
            )
        )

    return eval_models
