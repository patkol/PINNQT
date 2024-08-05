# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Dict, Any
import torch

from kolpinn import mathematics


# General
simulated_device_name = "barrier1"
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
n_hidden_layers = 9
n_neurons_per_hidden_layer = 100
activation_function = torch.nn.Tanh()
model_dtype = torch.float32

# Training
max_n_training_steps = None
max_time = None
min_loss = 1000e-6
report_each = 1
Optimizer = torch.optim.LBFGS
optimizer_kwargs = {"lr": 1, "tolerance_grad": 0, "tolerance_change": 0}
optimizer_reset_tol = 100
Scheduler = None
scheduler_kwargs: Dict[str, Any] = {}
loss_function = mathematics.complex_abs2
loss_aggregate_function = sum
# loss_function = lambda x: mathematics.complex_abs2(x)**2
# loss_aggregate_function = lambda losses: torch.sqrt(sum(losses))
fd_first_derivatives = True
fd_second_derivatives = True
# Whether the voltage/energy is an input to the NN
continuous_voltage = True
continuous_energy = True
# Whether to use the weights from the previous energy step
continuous_training = loaded_parameters_index is None
batch_sizes: Dict[str, int] = {
    "x": 20,
    "DeltaE": 4,
}

# Resolution
N_x = 50

# Plotting
plot_each_voltage = 10
plot_each_energy = 20
extra_plots = False
