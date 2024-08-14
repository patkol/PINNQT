# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import torch

import parameters as params
from physical_constants import NM, M_E, EPSILON_0, EV, CM, H_BAR, K_B


# Constants

BETA = 1 / (K_B * params.TEMPERATURE)


# Devices
# _no_tails: minimal
# _extended: there is a contact layer on each side with no potential drop
# If 'includes_contacts' is True the potential drop does
# not affect the outermost layers

device_kwargs_dict: dict[str, dict] = {
    "free1": {
        "boundaries": [0, 5 * NM],
        "potentials": [0, 0, 0],
        "m_effs": [M_E, M_E, M_E],
        "dopings": [0, 0, 0],
        "permittivities": [EPSILON_0, EPSILON_0, EPSILON_0],
        "includes_contacts": False,
    },
    "free2": {
        "boundaries": [0, 5 * NM, 10 * NM],
        "potentials": [0, 0, 0, 0],
        "m_effs": [M_E, M_E, M_E, M_E],
        "dopings": [0, 0, 0, 0],
        "permittivities": [EPSILON_0, EPSILON_0, EPSILON_0, EPSILON_0],
        "includes_contacts": False,
    },
    "barrier1": {
        "boundaries": [
            0 * NM,
            15 * NM,
            19 * NM,
            34 * NM,
        ],
        "potentials": [
            0 * EV,
            0 * EV,
            0.3 * EV,
            0 * EV,
            0 * EV,
        ],
        "m_effs": [
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            0,
            0,
            0,
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            12 * EPSILON_0,
            6 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
        "includes_contacts": False,
    },
    "barrier1_no_tails": {
        "boundaries": [0, 5 * NM],
        "potentials": [0, 0.3 * EV, 0],
        "m_effs": [0.065 * M_E, 0.1 * M_E, 0.065 * M_E],
        "dopings": [0, 0, 0],
        "permittivities": [12 * EPSILON_0, 11 * EPSILON_0, 12 * EPSILON_0],
        "includes_contacts": False,
    },
    "barrier1_extended": {
        "boundaries": [
            0 * NM,
            30 * NM,
            45 * NM,
            49 * NM,
            64 * NM,
            94 * NM,
        ],
        "potentials": [
            0 * EV,
            0 * EV,
            0 * EV,
            0.3 * EV,
            0 * EV,
            0 * EV,
            0 * EV,
        ],
        "m_effs": [
            0.065 * M_E,
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            1e19 / CM**3,
            0,
            0,
            0,
            1e19 / CM**3,
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
            6 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
        "includes_contacts": True,
    },
    "rtd1": {
        "boundaries": [0 * NM, 10 * NM, 12.6 * NM, 16.6 * NM, 19.2 * NM, 29.2 * NM],
        "potentials": [0 * EV, 0 * EV, 0.3 * EV, 0 * EV, 0.3 * EV, 0 * EV, 0 * EV],
        "m_effs": [
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        "dopings": [1e19 / CM**3, 0, 0, 1e18 / CM**3, 0, 0, 1e19 / CM**3],
        "permittivities": [
            12 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
        "includes_contacts": False,
    },
    "rtd1_no_tails": {
        "boundaries": [0 * NM, 2.6 * NM, 6.6 * NM, 9.2 * NM],
        "potentials": [0, 0.3 * EV, 0, 0.3 * EV, 0],
        "m_effs": [
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
        ],
        "dopings": [0, 0, 1e18 / CM**3, 0, 0],
        "permittivities": [
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
        ],
        "includes_contacts": False,
    },
    "rtd1_extended": {
        "boundaries": [
            0 * NM,
            30 * NM,
            45 * NM,
            47.6 * NM,
            51.6 * NM,
            54.2 * NM,
            69.2 * NM,
            99.2 * NM,
        ],
        "potentials": [
            0 * EV,
            0 * EV,
            0 * EV,
            0.3 * EV,
            0 * EV,
            0.3 * EV,
            0 * EV,
            0 * EV,
            0 * EV,
        ],
        "m_effs": [
            0.065 * M_E,
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            1e19 / CM**3,
            0,
            0,
            1e18 / CM**3,
            0,
            0,
            1e19 / CM**3,
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
        "includes_contacts": True,
    },
}

device_kwargs = device_kwargs_dict[params.simulated_device_name]


def k_function(m, E):
    k_squared = 2 * m * E / H_BAR**2
    k_squared = k_squared.to(params.si_complex_dtype)

    return torch.sqrt(k_squared)
