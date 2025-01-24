import einops
import numpy as np
import torch
import torch.nn as nn
from .mappings import Mapping


class tokenizer(nn.Module):
    def __init__(self, env_name, output_dim, output_activation=None, device='cuda'):
        super(tokenizer, self).__init__()
        
        self.mapping = Mapping(env_name)
        self.map = self.mapping.get_map()
        
        
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.device = device
        
        self.zero_token = nn.Parameter(torch.randn(1, 1, output_dim))
        
        base = lambda input_dim: nn.Sequential(nn.Linear(input_dim, output_dim))
        self.tokenizers = torch.nn.ModuleDict()
        for key in self.map.keys():
            self.tokenizers[key] = base(len(self.map[key][0]))
            if output_activation is not None:
                self.tokenizers[key] = nn.Sequential(self.tokenizers[key], output_activation)
    
    def forward(self, x):
        x = self.mapping.create_observation(x)
        tokens = []
        for key in x.keys():
            inputs = x[key].to(self.device)
            if inputs.shape[-1] == 0:
                tokens.append(self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(1))
            else:
                tokens.append(self.tokenizers[key](inputs).unsqueeze(1))
        tokens = torch.cat(tokens, dim=1)
        return tokens

class detokenizer(nn.Module):
    def __init__(self, env_name, embedding_dim, action_dim, num_layers=1, global_input=False, output_activation=None, device='cuda'):
        super(detokenizer, self).__init__()
        
        self.mapping = Mapping(env_name)
        self.map = self.mapping.get_map()
        
        self.nbodies = len(self.map.keys())
        
        self.embedding_dim = embedding_dim 
        self.action_dim = action_dim
        self.ouptut_activation = output_activation
        
        self.device = device
        
        base = lambda input_dim: nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = base(action_dim)
            if output_activation is not None:
                self.detokenizers['global'] = nn.Sequential(self.detokenizers['global'], output_activation)
        else:
            for key in self.map.keys():
                self.detokenizers[key] = base(len(self.map[key][1]))
                if output_activation is not None:
                    self.detokenizers[key] = nn.Sequential(self.detokenizers[key], output_activation)
    
    def forward(self, x, weights=None):
        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x.to(self.device))
        
        action  = torch.zeros(x.shape[0], self.action_dim).to(self.device)
        
        for i, key in enumerate(self.map.keys()):
            current_action = self.detokenizers[key](x[:, i, :])
            action[:, self.map[key][1]] = current_action
        return action

    def weighted_sum(self, x, weights):
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

class Transformer(nn.Module):
    def __init__(self, nbodies, input_dim, dim_feedforward=256, nhead=6, nlayers=3, use_positional_encoding=False, device='cuda'):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.output_dim = input_dim
        self.use_positional_encoding = use_positional_encoding
        
        if use_positional_encoding:
            print("Using positional encoding")
            self.emboed_absolute_position = nn.Embedding(nbodies, input_dim)
        
    def forward(self, x):
        if self.use_positional_encoding:
            _, nbodies, _ = x.shape
            limb_indicies = torch.arange(0, nbodies).to(x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indicies)
            x = x + limb_idx_embedding
        x = self.encoder(x)
        return x
        
        
        
class BodyTransformer(Transformer):
    def __init__(self, nbodies, env_name, input_dim, dim_feedforward=256, nhead=6, nlayers=3, is_mixed = True, use_positional_encoding=False, first_hard_layer=1, random_mask=False, device='cuda'):
        super().__init__(nbodies, input_dim, dim_feedforward, nhead, nlayers, use_positional_encoding, device)
        self.mapping = Mapping(env_name)
        shortest_path_matrix = self.mapping.shortest_path_matrix.to(device)
        adjacency_matrix = shortest_path_matrix < 2
        
        if random_mask:
            num_nozero = torch.sum(adjacency_matrix) - adjacency_matrix.shape[0]
            prob_nonzero  = num_nozero / torch.numel(adjacency_matrix)
            adjacency_matrix =  torch.rand(adjacency_matrix.shape) > prob_nonzero
            adjacency_matrix.fill_diagonal_(True)
        
        self.nbodies = adjacency_matrix.shape[0]
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.is_mixed = is_mixed
        self.use_positional_encoding = use_positional_encoding
        self.first_hard_layer = first_hard_layer
        
        self.register_buffer('adjacency_matrix', adjacency_matrix)
        
    def forward(self, x):
        if self.use_positional_encoding:
            _, nbodies, _ = x.shape
            limb_indicies = torch.arange(0, nbodies).to(x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indicies)
            x = x + limb_idx_embedding
        x = self.encoder(x, mask=~self.adjacency_matrix, is_mixed=self.is_mixed, return_intermediate=False, first_hard_layer=self.first_hard_layer)
        return x
        