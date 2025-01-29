import torch
import torch.nn as nn
from .body_transformer import tokenizer, detokenizer

class BodyNet(nn.Module):
    def __init__(self, env_name, net, action_dim, embedding_dim, output_activation=None, global_input=False, fixed_std=0.1, device='cuda'):
        super(BodyNet, self).__init__()
        
        self.std = fixed_std
        self.state_dim = action_dim
        
        self.tokenizer = tokenizer(env_name, embedding_dim, output_activation=output_activation, device=device)
        self.net = net
        self.detokenizer = detokenizer(env_name, embedding_dim, action_dim, device=device, global_input=global_input)
        
        self.tokenizer.to(device)
        self.net.to(device)
        self.detokenizer.to(device)
    
    def forward(self, x, time, context):
        x = self.tokenizer(x)  # Tokenize input
        x = self.net(x, time, context)  # Pass through BodyTransformer
        x = self.detokenizer(x)  # Detokenize to original shape
        return x
    
    def mode(self, x):
        return self.forward(x)

    def log_prob(self, x, action):
        mu = self.forward(x)
        std = self.std
        
        return torch.distributions.Normal(mu, std).log_prob(action).sum(dim=1)