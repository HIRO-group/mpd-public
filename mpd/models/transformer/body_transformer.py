import einops
import numpy as np
import torch
from .mappings import Mapping
import einops
from mpd.models.layers.layers import TimeEncoder

import torch.nn as nn


class tokenizer(nn.Module):
    def __init__(self, env_name, output_dim, output_activation=None, device='cuda'):
        """
        Tokenizer module for transforming input data into tokens.
        
        Args:
            env_name (str): Name of the environment.
            output_dim (int): Dimension of the output tokens.
            output_activation (nn.Module, optional): Activation function applied to the output tokens. Defaults to None.
            device (str, optional): Device to use for computation. Defaults to 'cuda'.
        """
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
        """
        Forward pass of the tokenizer module.
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, time_steps, state_dim).
        
        Returns:
            torch.Tensor: Tokens with shape (batch_size, time_steps, nbodies, token_dim).
        """
        # Create a dictionary of mapped observations
        x = self.mapping.create_observation(x)
        
        # Process tokens for each body part
        tokens = []
        for key in x.keys():
            inputs = x[key].to(self.device)  # Shape: (batch_size, time_steps, mapped_dim)
            
            if inputs.shape[-1] == 0:
                # If no inputs, use zero tokens
                tokens.append(
                    self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(2)
                )  # Add nbodies dimension
            else:
                # Tokenize the inputs for this body
                tokens.append(
                    self.tokenizers[key](inputs).unsqueeze(2)
                )  # Add nbodies dimension
        
        # Concatenate along the nbodies dimension
        tokens = torch.cat(tokens, dim=2)  # Shape: (batch_size, time_steps, nbodies, token_dim)
        return tokens


class detokenizer(nn.Module):
    def __init__(self, env_name, embedding_dim, action_dim, n_points_in_traj=64, num_layers=1, global_input=False, output_activation=None, device='cuda'):
        """
        Detokenizer module for reconstructing the original input format.
        
        Args:
            env_name (str): Name of the environment.
            embedding_dim (int): Dimension of the token embeddings.
            action_dim (int): Dimension of the output action.
            n_points_in_traj (int): Number of trajectory points (time steps).
            num_layers (int, optional): Number of layers. Defaults to 1.
            global_input (bool, optional): Whether to use a global input for detokenization. Defaults to False.
            output_activation (nn.Module, optional): Activation function applied to the output action. Defaults to None.
            device (str, optional): Device for computation. Defaults to 'cuda'.
        """
        super(detokenizer, self).__init__()
        
        self.n_points_in_traj = n_points_in_traj  # 64 time steps
        self.mapping = Mapping(env_name)
        self.map = self.mapping.get_map()
        
        self.nbodies = len(self.map.keys())  # Number of parts (1 for point robot, 7 for Panda)
        self.embedding_dim = embedding_dim  
        self.action_dim = action_dim  # Should match `state_dim = 4`
        self.output_activation = output_activation
        self.device = device
        
        base = lambda input_dim: nn.Linear(embedding_dim, input_dim)
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

    def forward(self, x):
        """
        Forward pass of the detokenizer.
        
        Args:
            x (torch.Tensor): Tokens with shape `(batch_size, time_steps, nbodies, token_dim)`.
        
        Returns:
            torch.Tensor: Reconstructed output of shape `(batch_size, time_steps, state_dim)`.
        """
        batch_size, time_steps, nbodies, token_dim = x.shape  # Expected shape: (32, 64, 1, 36)
        
        # Initialize reconstructed action with correct shape (batch_size, time_steps, state_dim)
        action = torch.zeros(batch_size, time_steps, self.action_dim, device=self.device)  # Shape: (32, 64, 4)

        for i, key in enumerate(self.map.keys()):
            current_action = self.detokenizers[key](x[:, :, i, :])  # Shape: (batch_size, time_steps, mapped_dim)
            
            # Ensure correct shape before assignment
            if current_action.dim() == 3 and current_action.shape[2] == 1:
                current_action = current_action.squeeze(-1)  # Remove singleton dim if present
            
            # Assign decoded values to the appropriate indices
            action[:, :, self.map[key][1]] = current_action  # Assign to state_dim positions

        # 🚀 ✅ Final shape should be `(batch_size, time_steps, state_dim)`
        return action


class Transformer(nn.Module):
    def __init__(self, nbodies, input_dim, dim_feedforward=256, nhead=6, nlayers=3, use_positional_encoding=False, device='cuda'):
        """
        Transformer module for processing input data.
        
        Args:
            nbodies (int): Number of bodies in the system.
            input_dim (int): Dimension of the input data.
            dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 256.
            nhead (int, optional): Number of attention heads. Defaults to 6.
            nlayers (int, optional): Number of transformer layers. Defaults to 3.
            use_positional_encoding (bool, optional): Whether to use positional encoding. Defaults to False.
            device (str, optional): Device to use for computation. Defaults to 'cuda'.
        """
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.output_dim = input_dim
        self.use_positional_encoding = use_positional_encoding
        
        if use_positional_encoding:
            print("Using positional encoding")
            self.emboed_absolute_position = nn.Embedding(nbodies, input_dim)
        
    def forward(self, x):
        """
        Forward pass of the transformer module.
        
        Args:
            x (torch.Tensor): Input data.
        
        Returns:
            torch.Tensor: Transformed output.
        """
        if self.use_positional_encoding:
            _, nbodies, _ = x.shape
            limb_indicies = torch.arange(0, nbodies).to(x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indicies)
            x = x + limb_idx_embedding
        x = self.encoder(x)
        return x
        
        
class BodyTransformer(Transformer):
    def __init__(self, nbodies, env_name, input_dim, dim_feedforward=256, nhead=6, nlayers=3, is_mixed=True, use_positional_encoding=False, first_hard_layer=1, random_mask=False, device='cuda'):
        """
        BodyTransformer module for processing input data specific to body systems.
        
        Args:
            nbodies (int): Number of bodies in the system.
            env_name (str): Name of the environment.
            input_dim (int): Dimension of the input data.
            dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 256.
            nhead (int, optional): Number of attention heads. Defaults to 6.
            nlayers (int, optional): Number of transformer layers. Defaults to 3.
            is_mixed (bool, optional): Whether to use mixed attention. Defaults to True.
            use_positional_encoding (bool, optional): Whether to use positional encoding. Defaults to False.
            first_hard_layer (int, optional): Index of the first hard layer. Defaults to 1.
            random_mask (bool, optional): Whether to use a random mask. Defaults to False.
            device (str, optional): Device to use for computation. Defaults to 'cuda'.
        """
        super().__init__(nbodies, input_dim, dim_feedforward, nhead, nlayers, use_positional_encoding, device)
        self.mapping = Mapping(env_name)
        shortest_path_matrix = self.mapping.shortest_path_matrix.to(device)
        adjacency_matrix = (shortest_path_matrix <= 2).float()  #check this at some point #shortest_path_matrix < 2
        
        if random_mask:
            num_nozero = torch.sum(adjacency_matrix) - adjacency_matrix.shape[0]
            prob_nonzero = num_nozero / torch.numel(adjacency_matrix)
            adjacency_matrix = torch.rand(adjacency_matrix.shape) > prob_nonzero
            adjacency_matrix.fill_diagonal_(True)
        
        self.nbodies = adjacency_matrix.shape[0]
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.is_mixed = is_mixed
        self.use_positional_encoding = use_positional_encoding
        self.first_hard_layer = first_hard_layer
        
        self.register_buffer('adjacency_matrix', adjacency_matrix)
        
        self.time_mlp = TimeEncoder(32, input_dim) # batch_size, time_emb_dim
        
    def forward(self, x, time, context):
        """
        Forward pass of the BodyTransformer module.

        Args:
            x (torch.Tensor): Input data of shape [batch_size, time_steps, nbodies, token_dim].
            time (torch.Tensor): Time information of shape [batch_size].
            context (torch.Tensor or None): Context information, if available.

        Returns:
            torch.Tensor: Transformed output.
        """
        batch_size, time_steps, nbodies, token_dim = x.shape

        # Positional encoding for bodies
        if self.use_positional_encoding:
            limb_indices = torch.arange(0, nbodies, device=x.device)
            limb_idx_embedding = self.embed_absolute_position(limb_indices)
            x = x + limb_idx_embedding

        # Time embedding
        t_emb = self.time_mlp(time)  # Shape: [batch_size, token_dim]
        t_emb = t_emb.unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, token_dim]

        # Context embedding (optional)
        if context is not None:
            c_emb = self.context_mlp(context)  # Shape: [batch_size, token_dim]
            c_emb = c_emb.unsqueeze(1).expand(-1, time_steps, -1)  # Shape: [batch_size, time_steps, token_dim]
            x = x + t_emb + c_emb
        else:
            x = x + t_emb

        # Flatten time_steps and nbodies into a single dimension
        x = x.view(batch_size, time_steps * nbodies, token_dim)  # Shape: [batch_size, seq_len, token_dim]

        expanded_adj_matrix = self.adjacency_matrix.repeat(time_steps, time_steps)  # Shape: (time_steps * nbodies, time_steps * nbodies)
        adjacency_mask = expanded_adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)

        # ✅ FIX: Ensure Transformer mask matches `num_heads`
        num_heads = self.encoder.layers[0].self_attn.num_heads
        adjacency_mask = adjacency_mask.repeat(num_heads, 1, 1)
        mask = (~adjacency_mask.bool()).to(torch.bool)
        x = self.encoder(x, mask=mask)

        # Reshape back to (batch_size, time_steps, nbodies, token_dim)
        x = x.view(batch_size, time_steps, nbodies, token_dim)

        return x
