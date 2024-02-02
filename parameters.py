import numpy as np
import torch

from kolpinn import mathematics


# General
simulated_device_name = 'rtd0_extended'
seed = 0
device = 'cuda'
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128

# Models
loaded_parameters_index = 215
# Whether to use the state of the saved optimizer (possibly overwriting optimizer_kwargs)
load_optimizer = False
load_scheduler = False
n_hidden_layers = 5
n_neurons_per_hidden_layer = 75
activation_function = torch.nn.Tanh()
model_dtype = torch.float32
complex_polar = False
model_ab = True # Otherwise phi is modelled directly

# Training
max_n_training_steps = 0
max_time = None
min_loss = 100e-6
report_each = 50
Optimizer = torch.optim.Adam
optimizer_kwargs = {'lr': 1e-5}
Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_kwargs = {'factor': 0.2, 'cooldown': 12, 'min_lr': 1e-6, 'eps': 0, 'verbose': True}
loss_function = mathematics.complex_abs2
fd_first_derivatives = True
fd_second_derivatives = True
# Whether to use the weights from the previous energy step
continuous_training = loaded_parameters_index is None

# Resolution
N_x_training = 500
N_x_validation = 500
batch_size_x = 500
dx = 1e-9 * 1e-2
