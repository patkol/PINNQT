# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import torch

from kolpinn import mathematics


# General
simulated_device_name = 'barrier1_extended'
seed = 0
device = 'cuda'
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128

# Models
loaded_parameters_index = 133
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
optimizer_kwargs = {'lr': 0.5, 'tolerance_grad': 0, 'tolerance_change': 0}
optimizer_reset_tol = 100
Scheduler = None
scheduler_kwargs = None
def loss_function(x): return mathematics.complex_abs2(x)
def loss_aggregate_function(losses): return sum(losses)
# def loss_function(x): return mathematics.complex_abs2(x)**2
# def loss_aggregate_function(losses): return torch.sqrt(sum(losses))
fd_first_derivatives = True
fd_second_derivatives = True
# Whether the voltage/energy is an input to the NN
continuous_voltage = True
continuous_energy = True
# Whether to use the weights from the previous energy step
continuous_training = loaded_parameters_index is None

# Resolution
N_x = 500
# TEMP: Batching hard disabled batch_size_x = -1 # -1 for disabling batching

# Plotting
plot_each_voltage = 10
plot_each_energy = 20
extra_plots = False
