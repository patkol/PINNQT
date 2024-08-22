# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Dict, Any
import numpy as np
import torch

from kolpinn import mathematics

import physical_constants as consts


# General
simulated_device_name = "free3"
seed = 0
device = "cuda"
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128

# Models
loaded_parameters_index = None
# `load_optimizer`: Whether to use the state of the saved optimizer
#                   (possibly overwriting optimizer_kwargs)
load_optimizer = False
load_scheduler = False
save_optimizer = False
n_hidden_layers = 5
n_neurons_per_hidden_layer = 50
activation_function = torch.nn.Tanh()
model_dtype = torch.float32

# Training
max_n_training_steps = 0
max_time = None
min_loss = 50e-6
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
fd_first_derivatives = True
fd_second_derivatives = True
# Whether the voltage/energy is an input to the NN
continuous_voltage = True
continuous_energy = True
batch_sizes: Dict[str, int] = {
    # "x": 1000,
    # "DeltaE": 100,
}

# Plotting
plot_each_voltage = 1
plot_each_energy = 12  # 500
extra_plots = True

# Resolution
N_x = 1000

# Physical
E_MIN = 1e-2 * consts.EV
E_STEP = 5e-3 * consts.EV
E_MAX = 1e-2 * consts.EV
E_MIN += 1e-6 * consts.EV  # Avoiding problems at E == V (sqrt(E-V)' not defined)
E_MAX += E_STEP / 2  # Making sure that E_MAX is used

VOLTAGE_MIN = 0.0 * consts.VOLT
VOLTAGE_STEP = 0.002 * consts.VOLT
VOLTAGE_MAX = 0.0 * consts.VOLT
VOLTAGE_MAX += VOLTAGE_STEP / 2  # Making sure that VOLTAGE_MAX is used

TEMPERATURE = 300 * consts.KELVIN

energy_smoothing_range = 0.05 * consts.EV
transition_distance = 0.5 * consts.NM
dx = 0.01 * consts.NM  # Used for derivatives

V_OOM = 0.3 * consts.EV
M_EFF_OOM = 0.1 * consts.M_E
K_OOM = np.sqrt(2 * M_EFF_OOM * V_OOM / consts.H_BAR**2)
CURRENT_CONTINUITY_OOM = K_OOM / M_EFF_OOM
PROBABILITY_CURRENT_OOM = consts.H_BAR * K_OOM / M_EFF_OOM
