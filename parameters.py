import numpy as np
import torch

from kolpinn import mathematics


# General
simulated_device_name = 'field_barrier1'
seed = 0
device = 'cuda'
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128

# Models
loaded_parameters_index = 24
n_hidden_layers = 5
n_neurons_per_hidden_layer = 30
activation_function = torch.nn.SiLU()
model_dtype = torch.float32

# Training
max_n_training_steps = None
max_time = 3600
min_loss = 100e-6
report_each = 50
Optimizer = torch.optim.Adam
optimizer_kwargs = {'lr': 1e-3}
Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_kwargs = {'factor': 0.5}
loss_function = lambda x: x.transform(mathematics.complex_abs2)
# Whether to use the weights from the previous energy step
continuous_training = loaded_parameters_index is None

# Resolution
N_x_training = 1000
N_x_validation = 81
batch_size_x = 100
