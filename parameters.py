# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from typing import Dict, Any
import numpy as np
import torch

from kolpinn import mathematics

import physical_constants as consts


# General
simulated_device_name = "InGaAs_transistor_x"
V_el_guess_type = "transistor"
V_el_guess_kwargs = {
    "x_gate_L": 20 * consts.NM,
    "x_gate_R": 35 * consts.NM,
    "ramp_size": 2 * consts.NM,
    "V_channel": 0.9626336 * consts.EV,  # voltage2 is subtracted from this TODO: move
}
# voltage2: a voltage applied in the middle of the device. The voltage to the left
# is still assumed to be zero and the one on the right to be given by voltage
use_voltage2 = True
# V_el_guess_type = "rtd"
# V_el_guess_kwargs = {
#     "x_L": 30 * consts.NM,
#     "x_R": (99.2 - 30) * consts.NM,
#     "dx_smoothing": 20 * consts.NM,
#     "y0": 0,
#     "y1": 0.2 * consts.EV,
# }
# use_voltage2 = False
seed = 0
device = "cuda"
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128
E_fermi_search_range = (0, 2 * consts.EV)

# Models
loaded_parameters_index = None
energy_cutoff_start = 0.8 * consts.EV
loaded_parameters_NR_step = 0
loaded_V_el_index = None
loaded_V_el_NR_step = loaded_parameters_NR_step
# use_V_el_new: Whether to use V_el_new from loaded_V_el_NR_step - 1
use_V_el_new = True
# `load_optimizer`: Whether to use the state of the saved optimizer
#                   (possibly overwriting optimizer_kwargs)
load_optimizer = False
load_scheduler = False
save_optimizer = False
n_hidden_layers = 12
n_neurons_per_hidden_layer = 350
activation_function = torch.nn.Tanh()
model_dtype = torch.float32

# Training
max_n_training_steps = None
max_time = None
min_loss = 10000e-9
report_each = 1
Optimizer = torch.optim.LBFGS
optimizer_kwargs = {"lr": 1, "tolerance_grad": 0, "tolerance_change": 0}
optimizer_reset_tol = 100
Scheduler = None
scheduler_kwargs: Dict[str, Any] = {}
loss_function = mathematics.complex_abs2
loss_aggregate_function = sum
# loss_function = lambda x: mathematics.complex_abs2(x) ** 2
# loss_aggregate_function = lambda losses: torch.sqrt(sum(losses))
use_j_loss = False
fd_first_derivatives = True
fd_second_derivatives = True
# Whether the voltage/energy is an input to the NN
continuous_voltage = True
continuous_energy = True
batch_sizes: Dict[str, int] = {
    # "x": 200,
    # "DeltaE": 100,
}
n_newton_raphson_steps = 1
newton_raphson_rate = 1
reset_weights_per_nr_step = False
force_unity_coeff = False
soft_bc = False
# soft_bc_output: if True, soft BC will be applied to the output contacts even if
# soft_bc is False
soft_bc_output = False
# hard_bc_direction:
# 1: Force BC from input to output contact
# -1: vice versa
# 0: no hard BC
hard_bc_dir = -1
use_phi_one = True
# hard_bc_without_phi_one: linear / multiply_linear / conjugate
hard_bc_without_phi_one = "conjugate"
learn_phi_prime = False
learn_phi_prime_polar = False
U_input_scale = 0.1 * consts.VOLT
E_input_scale = 0.2 * consts.EV
E_input_scale_sqrt = None  # 2e-2 * consts.EV
x_input_scale = 10 * consts.NM

# Plotting
plot_each_voltage = 1
plot_each_energy = 200
extra_plots = True

# Physical
VOLTAGE_MIN = 0.6 * consts.VOLT
VOLTAGE_STEP = 0.05 * consts.VOLT
VOLTAGE_MAX = 0.60001 * consts.VOLT

VOLTAGE2_MIN = 0.0 * consts.VOLT
VOLTAGE2_STEP = 0.05 * consts.VOLT
VOLTAGE2_MAX = 0.00001 * consts.VOLT

E_MIN = 0.6 * consts.EV  # 1e-3 * consts.EV
E_STEP = 1e-3 * consts.EV
E_MAX = 1.5 * consts.EV
# E_MIN = 0.05 * consts.EV
# E_STEP = 0.05 * consts.EV
# E_MAX = 0.05 * consts.EV
E_MIN += 1e-6 * consts.EV  # Avoiding problems at E == V (sqrt(E-V)' not defined)

X_STEP = 0.05 * consts.NM

TEMPERATURE = 300 * consts.KELVIN

# CONSTANT_FERMI_LEVEL: None to find the correct fermi level.
#                       V_int and V_el are added.
CONSTANT_FERMI_LEVEL = 0.9206335951628302 * consts.EV  # 0.258 * consts.EV

dx = 0.01 * consts.NM  # Used for derivatives
dV_poisson = 1e-4 * consts.EV
energy_cutoff_delta = 0.2 * consts.EV
device_smoothing_distance = 4 * consts.NM

"""
ansatz: determines how a/b_phase are calculated.
  none: Unity
  wkb: WKB solution
  half_wkb: a (input-output) is WKB, b unity, swapped if hard_bc_dir == -1
      This makes sure that both ansaetze approach (about) unity at the boundary where we
      force the BC, such that the solution does not explode for highly negative
      energies / thick barriers.
"""
ansatz = "wkb"
ignore_wkb_phase = False  # Whether to use the absolute value as the wkb ansatz
"""
output_trafo: Transformation of the NN output a/b_output
    none: None
    scaled_exp: exp w/ imaginary part multiplied by K_OOM * x
    Veff: interpret output as an effective potential
"""
output_trafo = "none"

V_OOM = 0.3 * consts.EV
M_EFF_OOM = 0.1 * consts.M_E
K_OOM = np.sqrt(2 * M_EFF_OOM * V_OOM / consts.H_BAR**2)
CURRENT_CONTINUITY_OOM = K_OOM / M_EFF_OOM
PROBABILITY_CURRENT_OOM = consts.H_BAR * K_OOM / M_EFF_OOM

BETA = 1 / (consts.K_B * TEMPERATURE)


assert hard_bc_dir in (1, -1, 0)
assert not (learn_phi_prime and learn_phi_prime_polar)
assert not (
    (learn_phi_prime or learn_phi_prime_polar) and not force_unity_coeff
), "Not implemented, see phi_trafo"
assert ansatz in ("none", "plane", "wkb", "half_wkb")
