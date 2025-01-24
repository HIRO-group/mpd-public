import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[1024, 1024], activation = nn.ReLU()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        self.layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*self.layers)
        
        self.outpu_dim = input_dim
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)