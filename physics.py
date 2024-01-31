import numpy as np
import torch

from kolpinn.grid_quantities import QuantitiesFactory

import parameters as params


# Constants

H_BAR = 1.054571817e-34 # J s
Q_E = 1.60217663e-19
EV = Q_E
M_E = 9.1093837e-31
NM = 1e-9

E_MIN = 0.2 * EV
E_STEP = 0.01 * EV
E_MAX = 0.60001 * EV

A_L = 1 # Amplitude of the wave incoming from the left
A_R = 0 # Amplitude of the wave incoming from the right


# Devices

device_kwargs_dict = {
    'barrier1': {
        'boundaries': [0, 5 * NM],
        'potentials': [0, 0.3 * EV, 0],
        'm_effs': [0.065 * M_E, 0.1 * M_E, 0.065 * M_E],
    },
    'field_barrier1': {
        'boundaries': [0, 15 * NM, 19 * NM, 34 * NM],
        'potentials': [
            0,
            lambda q: -0.1 * EV * q['x'] / (34 * NM),
            lambda q: -0.1 * EV * q['x'] / (34 * NM) + 0.3 * EV,
            lambda q: -0.1 * EV * q['x'] / (34 * NM),
            -0.1 * EV,
        ],
        'm_effs': [
            0.065 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
    },
    # Same as 1, but using one NN only
    'field_barrier2': {
        'boundaries': [0, 34 * NM],
        'potentials': [
            0,
            lambda q: (-0.1 * EV * q['x'] / (34 * NM)
                       + 0.3 * EV * (q['x'] > 15*NM) * (q['x'] < 19*NM)),
            -0.1 * EV,
        ],
        'm_effs': [
            0.065 * M_E,
            lambda q: (0.065 * M_E
                       + 0.035 * M_E * (q['x'] > 15*NM) * (q['x'] < 19*NM)),
            0.065 * M_E,
        ],
    },
    'rtd0': {
        'boundaries': [0 * NM, 2.6 * NM, 6.6 * NM, 9.2 * NM],
        'potentials': [0, 0.3 * EV, 0, 0.3 * EV, 0],
        'm_effs': [
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
            0.1 * M_E,
            0.065 * M_E,
        ],
    },
    'rtd0_extended': {
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
    },
}

device_kwargs = device_kwargs_dict[params.simulated_device_name]


def k_function(m, E):
    k_squared = 2 * m * E / H_BAR**2
    if torch.is_tensor(k_squared):
        k_squared = k_squared.to(params.si_complex_dtype)
    else:
        k_squared = k_squared.set_dtype(params.si_complex_dtype)

    return torch.sqrt(k_squared)


# Device-dependent constants

V_OOM = 0.3 * EV  #abs(max(device_kwargs['potentials'], key=abs))
M_EFF_OOM = 0.1 * M_E  #abs(max(device_kwargs['m_effs'], key=abs))
K_OOM = np.sqrt(2 * M_EFF_OOM * V_OOM / H_BAR**2)
CURRENT_CONTINUITY_OOM = K_OOM / M_EFF_OOM
PROBABILITY_CURRENT_OOM = H_BAR / M_EFF_OOM  * K_OOM


quantities_factory = QuantitiesFactory()


