import numpy as np
import torch

from kolpinn import mathematics


# General
simulated_device_name = 'barrier1'
seed = 0
device = 'cuda'
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128

# Models
loaded_parameters_index = None
# Whether to use the state of the saved optimizer (possibly overwriting optimizer_kwargs)
load_optimizer = True
load_scheduler = True
n_hidden_layers = 5
n_neurons_per_hidden_layer = 30
activation_function = torch.nn.SiLU()
model_dtype = torch.float32
complex_polar = False

# Training
max_n_training_steps = None
max_time = None
min_loss = 1000e-6
report_each = 50
Optimizer = torch.optim.Adam
optimizer_kwargs = {'lr': 1e-3}
Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_kwargs = {'factor': 0.2, 'cooldown': 12, 'min_lr': 1e-6, 'eps': 0, 'verbose': True}
loss_function = mathematics.complex_abs2
fd_first_derivatives = False # Except for boundaries
fd_second_derivatives = False
# Whether to use the weights from the previous energy step
continuous_training = loaded_parameters_index is None

# Resolution
N_x_training = 500
N_x_validation = 500
batch_size_x = 500
