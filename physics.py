# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import numpy as np
import torch

import parameters as params


# Constants

METER_SI = 1
CM_SI = 1e-2
NM_SI = 1e-9
SECOND_SI = 1
Q_E_SI = 1.60217663e-19
EV_SI = Q_E_SI
M_E_SI = 9.1093837e-31
VOLT_SI = 1
KELVIN_SI = 1
K_B_SI = 1.38064852e-23
H_BAR_SI = 1.054571817e-34  # J s
H_SI = H_BAR_SI * 2 * np.pi
EPSILON_0_SI = 8.8541878128e-12

NM = 1
Q_E = 1
EV = 1
M_E = 1

DISTANCE_SCALING = NM / NM_SI
CHARGE_SCALING = Q_E / Q_E_SI
ENERGY_SCALING = EV / EV_SI
MASS_SCALING = M_E / M_E_SI
METER = METER_SI * DISTANCE_SCALING
CM = CM_SI * DISTANCE_SCALING
TIME_SCALING = np.sqrt(MASS_SCALING * DISTANCE_SCALING**2 / ENERGY_SCALING)
SECOND = SECOND_SI * TIME_SCALING
VOLTAGE_SCALING = ENERGY_SCALING / CHARGE_SCALING
VOLT = VOLT_SI * VOLTAGE_SCALING
K_B_SCALING = DISTANCE_SCALING**2 * MASS_SCALING / TIME_SCALING**2
K_B = K_B_SI * K_B_SCALING
KELVIN = KELVIN_SI * ENERGY_SCALING / K_B_SCALING
ACTION_SCALING = ENERGY_SCALING * TIME_SCALING
H_BAR = H_BAR_SI * ACTION_SCALING
H = H_SI * ACTION_SCALING
EPSILON_0 = EPSILON_0_SI * CHARGE_SCALING / VOLTAGE_SCALING / DISTANCE_SCALING


E_MIN = 5e-4 * EV
E_STEP = 2e-4 * EV
E_MAX = 0.4 * EV
E_MIN += 1e-6 * EV  # Avoiding problems at E == V (sqrt(E-V)' not defined)
E_MAX += E_STEP / 2  # Making sure that E_MAX is used

VOLTAGE_MIN = 0.0 * VOLT
VOLTAGE_STEP = 0.002 * VOLT
VOLTAGE_MAX = 0.0 * VOLT
# VOLTAGE_MIN = 0.125 * VOLT
# VOLTAGE_STEP = 0.05 * VOLT
# VOLTAGE_MAX = 0.275 * VOLT
VOLTAGE_MAX += VOLTAGE_STEP / 2  # Making sure that VOLTAGE_MAX is used

TEMPERATURE = 300 * KELVIN
BETA = 1 / (K_B * TEMPERATURE)

energy_smoothing_range = 0.05 * EV
transition_distance = 0.5 * NM
dx = 0.01 * NM  # Used for derivatives


# Devices
# _no_tails: minimal
# _extended: there is a contact layer on each side with no potential drop
# If 'includes_contacts' is True the potential drop does
# not affect the outermost layers

device_kwargs_dict: dict[str, dict] = {
    'free1': {
        'boundaries': [0, 5 * NM],
        'potentials': [0, 0, 0],
        'm_effs': [M_E, M_E, M_E],
        'dopings': [0, 0, 0],
        'permittivities': [EPSILON_0, EPSILON_0, EPSILON_0],
        'includes_contacts': False,
    },
    'free2': {
        'boundaries': [0, 5 * NM, 10 * NM],
        'potentials': [0, 0, 0, 0],
        'm_effs': [M_E, M_E, M_E, M_E],
        'dopings': [0, 0, 0, 0],
        'permittivities': [EPSILON_0, EPSILON_0, EPSILON_0, EPSILON_0],
        'includes_contacts': False,
    },
    'barrier1': {
        'boundaries': [
            0 * NM,
            15 * NM,
            19 * NM,
            34 * NM,
        ],
        'potentials': [
            0 * EV,
            0 * EV,
            0.3 * EV,
            0 * EV,
            0 * EV,
        ],
        'm_effs': [
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        'dopings': [
            1e19 / CM**3,
            0,
            0,
            0,
            1e19 / CM**3,
        ],
        'permittivities': [
            12 * EPSILON_0,
            12 * EPSILON_0,
            6 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
        'includes_contacts': False,
    },
    'barrier1_no_tails': {
        'boundaries': [0, 5 * NM],
        'potentials': [0, 0.3 * EV, 0],
        'm_effs': [0.065 * M_E, 0.1 * M_E, 0.065 * M_E],
        'dopings': [0, 0, 0],
        'permittivities': [12 * EPSILON_0, 11 * EPSILON_0, 12 * EPSILON_0],
        'includes_contacts': False,
    },
    'barrier1_extended': {
        'boundaries': [
            0 * NM,
            30 * NM,
            45 * NM,
            49 * NM,
            64 * NM,
            94 * NM,
        ],
        'potentials': [
            0 * EV,
            0 * EV,
            0 * EV,
            0.3 * EV,
            0 * EV,
            0 * EV,
            0 * EV,
        ],
        'm_effs': [
            0.065 * M_E,
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        'dopings': [
            1e19 / CM**3,
            1e19 / CM**3,
            0,
            0,
            0,
            1e19 / CM**3,
            1e19 / CM**3,
        ],
        'permittivities': [
            12 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
            6 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
        'includes_contacts': True,
    },
    'rtd1': {
        'boundaries': [0 * NM, 10 * NM, 12.6 * NM, 16.6 * NM, 19.2 * NM, 29.2 * NM],
        'potentials': [0 * EV, 0 * EV, 0.3 * EV, 0 * EV, 0.3 * EV, 0 * EV, 0 * EV],
        'm_effs': [
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        'dopings': [1e19 / CM**3, 0, 0, 1e18 / CM**3, 0, 0, 1e19 / CM**3],
        'permittivities': [
            12 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
        'includes_contacts': False,
    },
    'rtd1_no_tails': {
        'boundaries': [0 * NM, 2.6 * NM, 6.6 * NM, 9.2 * NM],
        'potentials': [0, 0.3 * EV, 0, 0.3 * EV, 0],
        'm_effs': [
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
        ],
        'dopings': [0, 0, 1e18 / CM**3, 0, 0],
        'permittivities': [
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
        ],
        'includes_contacts': False,
    },
    'rtd1_extended': {
        'boundaries': [
            0 * NM,
            30 * NM,
            45 * NM,
            47.6 * NM,
            51.6 * NM,
            54.2 * NM,
            69.2 * NM,
            99.2 * NM,
        ],
        'potentials': [
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
        'm_effs': [
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
        'dopings': [
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
        'permittivities': [
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
        'includes_contacts': True,
    },
}

device_kwargs = device_kwargs_dict[params.simulated_device_name]


def k_function(m, E):
    k_squared = 2 * m * E / H_BAR**2
    k_squared = k_squared.to(params.si_complex_dtype)

    return torch.sqrt(k_squared)


# Device-dependent constants

V_OOM = 0.3 * EV
M_EFF_OOM = 0.1 * M_E
K_OOM = np.sqrt(2 * M_EFF_OOM * V_OOM / H_BAR**2)
CURRENT_CONTINUITY_OOM = K_OOM / M_EFF_OOM
PROBABILITY_CURRENT_OOM = H_BAR * K_OOM / M_EFF_OOM
