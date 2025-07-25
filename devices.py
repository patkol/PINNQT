# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from physical_constants import NM, M_E, EPSILON_0, EV, CM
import parameters as params
import formulas


# Devices
# _no_tails: minimal
# _extended: there is a contact layer on each side with no potential drop

device_kwargs_dict: dict[str, dict] = {
    "free1": {
        "boundaries": [0, 5 * NM],
        "potentials": [0, 0, 0],
        "m_effs": [M_E, M_E, M_E],
        "dopings": [0, 0, 0],
        "permittivities": [EPSILON_0, EPSILON_0, EPSILON_0],
    },
    "free2": {
        "boundaries": [0, 5 * NM, 10 * NM],
        "potentials": [0, 0, 0, 0],
        "m_effs": [M_E, M_E, M_E, M_E],
        "dopings": [0, 0, 0, 0],
        "permittivities": [EPSILON_0, EPSILON_0, EPSILON_0, EPSILON_0],
    },
    "free3": {
        "boundaries": [0, 100 * NM],
        "potentials": [0, 0, 0],
        "m_effs": [M_E, M_E, M_E],
        "dopings": [0, 0, 0],
        "permittivities": [EPSILON_0, EPSILON_0, EPSILON_0],
    },
    "free4": {
        "boundaries": [0, 15 * NM],
        "potentials": [0, 0, 0],
        "m_effs": [M_E, M_E, M_E],
        "dopings": [0, 0, 0],
        "permittivities": [EPSILON_0, EPSILON_0, EPSILON_0],
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
    },
    "barrier1_no_tails": {
        "boundaries": [0, 4 * NM],
        "potentials": [0, 0.3 * EV, 0],
        "m_effs": [0.065 * M_E, 0.1 * M_E, 0.065 * M_E],
        "dopings": [0, 0, 0],
        "permittivities": [12 * EPSILON_0, 11 * EPSILON_0, 12 * EPSILON_0],
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
    },
    "barrier1_extended_3layers": {
        "boundaries": [
            0 * NM,
            45 * NM,
            49 * NM,
            94 * NM,
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
            lambda q: 1e19 / CM**3 * (q["x"] < 30 * NM),
            0,
            lambda q: 1e19 / CM**3 * (q["x"] >= 64 * NM),
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            12 * EPSILON_0,
            6 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
    },
    "barrier1_extended_combined": {
        "boundaries": [
            0 * NM,
            94 * NM,
        ],
        "potentials": [
            0 * EV,
            lambda q: 0.3 * EV * (q["x"] >= 45 * NM) * (q["x"] < 49 * NM),
            0 * EV,
        ],
        "m_effs": [
            0.065 * M_E,
            lambda q: 0.065 * M_E
            + 0.035 * M_E * (q["x"] >= 45 * NM) * (q["x"] < 49 * NM),
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            lambda q: 1e19 / CM**3 * ((q["x"] < 30 * NM) + (q["x"] >= 64 * NM)),
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            lambda q: 12 * EPSILON_0
            - 6 * EPSILON_0 * (q["x"] >= 45 * NM) * (q["x"] < 49 * NM),
            12 * EPSILON_0,
        ],
    },
    "barrier1_combined_smoothed": {
        "boundaries": [
            0 * NM,
            34 * NM,
        ],
        "potentials": [
            0 * EV,
            lambda q: 0.3
            * EV
            * (
                formulas.smooth_transition(
                    q["x"],
                    x0=15 * NM - params.device_smoothing_distance / 2,
                    x1=15 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
                - formulas.smooth_transition(
                    q["x"],
                    x0=19 * NM - params.device_smoothing_distance / 2,
                    x1=19 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
            ),
            0 * EV,
        ],
        "m_effs": [
            0.065 * M_E,
            lambda q: 0.065 * M_E
            + 0.035
            * M_E
            * (
                formulas.smooth_transition(
                    q["x"],
                    x0=15 * NM - params.device_smoothing_distance / 2,
                    x1=15 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
                - formulas.smooth_transition(
                    q["x"],
                    x0=19 * NM - params.device_smoothing_distance / 2,
                    x1=19 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
            ),
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            0,
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            lambda q: 12 * EPSILON_0
            - 6 * EPSILON_0 * (q["x"] >= 15 * NM) * (q["x"] < 19 * NM),
            12 * EPSILON_0,
        ],
    },
    "barrier1_extended_combined_smoothed": {
        "boundaries": [
            0 * NM,
            94 * NM,
        ],
        "potentials": [
            0 * EV,
            lambda q: 0.3
            * EV
            * (
                formulas.smooth_transition(
                    q["x"],
                    x0=45 * NM - params.device_smoothing_distance / 2,
                    x1=45 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
                - formulas.smooth_transition(
                    q["x"],
                    x0=49 * NM - params.device_smoothing_distance / 2,
                    x1=49 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
            ),
            0 * EV,
        ],
        "m_effs": [
            0.065 * M_E,
            lambda q: 0.065 * M_E
            + 0.035
            * M_E
            * (
                formulas.smooth_transition(
                    q["x"],
                    x0=45 * NM - params.device_smoothing_distance / 2,
                    x1=45 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
                - formulas.smooth_transition(
                    q["x"],
                    x0=49 * NM - params.device_smoothing_distance / 2,
                    x1=49 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
            ),
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            lambda q: 1e19 / CM**3 * ((q["x"] < 30 * NM) + (q["x"] >= 64 * NM)),
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            lambda q: 12 * EPSILON_0
            - 6 * EPSILON_0 * (q["x"] >= 45 * NM) * (q["x"] < 49 * NM),
            12 * EPSILON_0,
        ],
    },
    # like matlab short_barrier, more heavily doped s.t. we can simulate shorter contacts
    "barrier2": {
        "boundaries": [
            0 * NM,
            16 * NM,
            31 * NM,
            35 * NM,
            50 * NM,
            66 * NM,
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
            5e19 / CM**3,
            5e19 / CM**3,
            0,
            0,
            0,
            5e19 / CM**3,
            5e19 / CM**3,
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
    },
    # barrier1 with constant eff. mass
    "barrier3": {
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
            0.065 * M_E,
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
    },
    "barrier3_combined_smoothed": {
        "boundaries": [
            0 * NM,
            34 * NM,
        ],
        "potentials": [
            0 * EV,
            lambda q: 0.3
            * EV
            * (
                formulas.smooth_transition(
                    q["x"],
                    x0=15 * NM - params.device_smoothing_distance / 2,
                    x1=15 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
                - formulas.smooth_transition(
                    q["x"],
                    x0=19 * NM - params.device_smoothing_distance / 2,
                    x1=19 * NM + params.device_smoothing_distance / 2,
                    y0=0,
                    y1=1,
                )
            ),
            0 * EV,
        ],
        "m_effs": [
            0.065 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            0,
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            lambda q: 12 * EPSILON_0
            - 6 * EPSILON_0 * (q["x"] >= 15 * NM) * (q["x"] < 19 * NM),
            12 * EPSILON_0,
        ],
    },
    # Large barrier
    "barrier4": {
        "boundaries": [
            0 * NM,
            25 * NM,
            75 * NM,
            100 * NM,
        ],
        "potentials": [
            0 * EV,
            0 * EV,
            1 * EV,
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
    },
    "rtd1_extended_5layers": {
        "boundaries": [
            0 * NM,
            45 * NM,
            47.6 * NM,
            51.6 * NM,
            54.2 * NM,
            99.2 * NM,
        ],
        "potentials": [
            0 * EV,
            0 * EV,
            0.3 * EV,
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
            0.1 * M_E,
            0.065 * M_E,
            0.065 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            lambda q: 1e19 / CM**3 * (q["x"] < 30 * NM),
            0,
            1e18 / CM**3,
            0,
            lambda q: 1e19 / CM**3 * (q["x"] > 69.2 * NM),
            1e19 / CM**3,
        ],
        "permittivities": [
            12 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            11 * EPSILON_0,
            12 * EPSILON_0,
            12 * EPSILON_0,
        ],
    },
    "rtd1_extended_5layers_real_params": {
        "boundaries": [
            0 * NM,
            45 * NM,
            47.6 * NM,
            51.6 * NM,
            54.2 * NM,
            99.2 * NM,
        ],
        "potentials": [
            0 * EV,
            0 * EV,
            0.251 * EV,
            0 * EV,
            0.251 * EV,
            0 * EV,
            0 * EV,
        ],
        "m_effs": [
            0.0635 * M_E,
            0.0635 * M_E,
            0.0895 * M_E,
            0.0635 * M_E,
            0.0895 * M_E,
            0.0635 * M_E,
            0.0635 * M_E,
        ],
        "dopings": [
            1e19 / CM**3,
            lambda q: 1e19 / CM**3 * (q["x"] < 30 * NM),
            0,
            0,
            0,
            lambda q: 1e19 / CM**3 * (q["x"] > 69.2 * NM),
            1e19 / CM**3,
        ],
        "permittivities": [
            12.9 * EPSILON_0,
            12.9 * EPSILON_0,
            12.0 * EPSILON_0,
            12.9 * EPSILON_0,
            12.0 * EPSILON_0,
            12.9 * EPSILON_0,
            12.9 * EPSILON_0,
        ],
    },
    # Corresponds to real params
    "InGaAs_transistor_x": {
        "boundaries": [
            0 * NM,
            55 * NM,
        ],
        "potentials": [
            0 * EV,
            0 * EV,
            0 * EV,
        ],
        "m_effs": [
            0.041 * M_E,
            0.041 * M_E,
            0.041 * M_E,
        ],
        "dopings": [  # Not accurate: not solving Poisson here, only used for fermi energy
            4e19 / CM**3,
            0 / CM**3,
            4e19 / CM**3,
        ],
        "permittivities": [  # vary along y
            0 * EPSILON_0,
            0 * EPSILON_0,
            0 * EPSILON_0,
        ],
    },
}
