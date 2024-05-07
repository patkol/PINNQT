# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import numpy as np
import torch

import parameters as params



# Constants

H_BAR = 1.054571817e-34 # J s
H = H_BAR * 2*np.pi
Q_E = 1.60217663e-19
EV = Q_E
M_E = 9.1093837e-31
NM = 1e-9
K_B = 1.38064852e-23
KELVIN = 1
VOLT = 1
CM = 1e-2

E_MIN = 5e-4 * EV
E_STEP = 5e-3 * EV
E_MAX = 0.4 * EV
E_MIN += 1e-6 * EV  # Avoiding problems at E == V (sqrt(E-V)' not defined)
E_MAX += E_STEP / 2  # Making sure that E_MAX is used

VOLTAGE_MIN = 0.15 * VOLT
VOLTAGE_STEP = 0.025 * VOLT
VOLTAGE_MAX = 0.25 * VOLT
VOLTAGE_MAX += VOLTAGE_STEP / 2  # Making sure that VOLTAGE_MAX is used

TEMPERATURE = 300 * KELVIN
BETA = 1 / (K_B * TEMPERATURE)

energy_smoothing_range = 0.05 * EV
transition_distance = 0.5 * NM
dx = 0.01 * NM  # Used for derivatives


# Devices

device_kwargs_dict = {
    'free1': {
        'boundaries': [0, 5 * NM],
        'potentials': [0, 0, 0],
        'm_effs': [M_E, M_E, M_E],
        'dopings': [0, 0, 0],
    },
    'free2': {
        'boundaries': [0, 5 * NM, 10 * NM],
        'potentials': [0, 0, 0, 0],
        'm_effs': [M_E, M_E, M_E, M_E],
        'dopings': [0, 0, 0, 0],
    },
    'barrier1': {
        'boundaries': [0, 5 * NM],
        'potentials': [0, 0.3 * EV, 0],
        'm_effs': [0.065 * M_E, 0.1 * M_E, 0.065 * M_E],
        'dopings': [0, 0, 0],
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
    },
}

device_kwargs = device_kwargs_dict[params.simulated_device_name]


def k_function(m, E):
    k_squared = 2 * m * E / H_BAR**2
    k_squared = k_squared.to(params.si_complex_dtype)

    return torch.sqrt(k_squared)


# Device-dependent constants

V_OOM = 0.3 * EV  #abs(max(device_kwargs['potentials'], key=abs))
M_EFF_OOM = 0.1 * M_E  #abs(max(device_kwargs['m_effs'], key=abs))
K_OOM = np.sqrt(2 * M_EFF_OOM * V_OOM / H_BAR**2)
CURRENT_CONTINUITY_OOM = K_OOM / M_EFF_OOM
PROBABILITY_CURRENT_OOM = H_BAR * K_OOM / M_EFF_OOM
