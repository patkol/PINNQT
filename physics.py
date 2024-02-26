import numpy as np
import torch

from kolpinn import mathematics
from kolpinn.grid_quantities import QuantitiesFactory

import parameters as params



# Constants

H_BAR = 1.054571817e-34 # J s
Q_E = 1.60217663e-19
EV = Q_E
M_E = 9.1093837e-31
NM = 1e-9

E_MIN = 0.4 * EV
E_STEP = 0.00001 * EV
E_MAX = 0.40000 * EV
E_MIN += 1e-6 * EV # Avoiding problems at E == V (sqrt(E-V)' not defined)
E_MAX += 2e-6 * EV # Making sure that E_MAX is used

A_L = 1 # Amplitude of the wave incoming from the left
B_R = 0 # Amplitude of the wave incoming from the right

smoothing_range = 0.05 * EV
transition_distance = 0.5 * NM


# Devices

# V_applied: Electron potential due to an applied voltage of `V` over a distance `d`
V_applied = lambda V, d, q: -V * EV * q['x'] / d

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
            lambda q: V_applied(0.1, 34 * NM, q),
            lambda q: V_applied(0.1, 34 * NM, q) + 0.3 * EV,
            lambda q: V_applied(0.1, 34 * NM, q),
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
            lambda q: (V_applied(0.1, 34 * NM, q)
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
    'rtd0.1': {
        'boundaries': [0 * NM, 10 * NM, 12.6 * NM, 16.6 * NM, 19.2 * NM, 29.2 * NM],
        'potentials': [
            0 * EV,
            lambda q: V_applied(0.1, 29.2 * NM, q),
            lambda q: V_applied(0.1, 29.2 * NM, q) + 0.3 * EV,
            lambda q: V_applied(0.1, 29.2 * NM, q),
            lambda q: V_applied(0.1, 29.2 * NM, q) + 0.3 * EV,
            lambda q: V_applied(0.1, 29.2 * NM, q),
            -0.1 * EV,
        ],
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
    k_squared = k_squared.to(params.si_complex_dtype)

    return torch.sqrt(k_squared)


# Device-dependent constants

V_OOM = 0.3 * EV  #abs(max(device_kwargs['potentials'], key=abs))
M_EFF_OOM = 0.1 * M_E  #abs(max(device_kwargs['m_effs'], key=abs))
K_OOM = np.sqrt(2 * M_EFF_OOM * V_OOM / H_BAR**2)
CURRENT_CONTINUITY_OOM = K_OOM / M_EFF_OOM
PROBABILITY_CURRENT_OOM = H_BAR / M_EFF_OOM  * K_OOM


quantities_factory = QuantitiesFactory()
