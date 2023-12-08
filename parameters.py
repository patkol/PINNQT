import numpy as np
import torch

from kolpinn import mathematics


# General
simulated_device_name = 'barrier1'
seed = 0
device = 'cuda'
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128

# Training
n_training_steps = 1000
report_each = 100
max_time = None
Optimizer = torch.optim.Adam
learn_rate = 1e-3
loss_function = lambda x: x.transform(mathematics.complex_abs2).mean()

# Models
loaded_parameters_index = None
n_hidden_layers = 5
n_neurons_per_hidden_layer = 30
activation_function = torch.nn.SiLU()
model_dtype = torch.float32

# Resolution
N_x_training = 1000
N_E_training = 200
N_x_validation = 81
N_E_validation = 1
batch_size_x = 100
batch_size_E = 1

