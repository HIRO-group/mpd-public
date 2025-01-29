import torch
DIMS = {
    'EnvSimple2D-RobotPointMass': 4,
    'EnvNarrowPassageDense2D-RobotPointMass': 4,
    'EnvDense2D-RobotPointMass': 4,
    'EnvSpehres3D-RobotPanda': 7
}


MAPS = {
    'EnvSimple2D-RobotPointMass': {
        'position': [[0, 1], [0, 1]] # observation and action
    },
    'EnvNarrowPassageDense2D-RobotPointMass': {
        'position': [[0, 1], [0, 1]] # observation and action
    },
    'EnvDense2D-RobotPointMass': {
        'position': [[0, 1], [0, 1]] # observation and action
    },
    'EnvSpehres3D-RobotPanda': {
        'joint_1': [[0], [0]],
        'joint_2': [[1], [1]],
        'joint_3': [[2], [2]],
        'joint_4': [[3], [3]],
        'joint_5': [[4], [4]],
        'joint_6': [[5], [5]],
        'joint_7': [[6], [6]],
    }
}

SP_MATRICES = {
    'EnvSimple2D-RobotPointMass': torch.Tensor(
        [[0]]
    ),
    'EnvNarrowPassageDense2D-RobotPointMass': torch.Tensor(
        [[0]]
    ),
    'EnvDense2D-RobotPointMass': torch.Tensor(
        [[0]]
    ),
    'EnvSpehres3D-RobotPanda': torch.Tensor(
        [[0, 1, 2, 3, 4, 5, 6], 
         [1, 0, 1, 2, 3, 4, 5],
         [2, 1, 0, 1, 2, 3, 4],
         [3, 2, 1, 0, 1, 2, 3],
         [4, 3, 2, 1, 0, 1, 2],
         [5, 4, 3, 2, 1, 0, 1],
         [6, 5, 4, 3, 2, 1, 0]]
    )
}

class Mapping:
    def __init__(self, env_name: str):
        self.dim = DIMS[env_name]
        self.map = MAPS[env_name]
        
        self.shortest_path_matrix = SP_MATRICES[env_name]
        
    def get_map(self):
        return self.map

    def create_observation(self, observation: torch.Tensor):
        """
        Processes the observation tensor to map it according to the body-specific mappings.
        
        Args:
            observation (torch.Tensor): Input observation of shape (batch_size, time_steps, state_dim).
        
        Returns:
            dict: A dictionary where each key corresponds to a body, and each value is the observation
                slice for that body with shape (batch_size, time_steps, mapped_dim).
        """
        batch_size, time_steps, state_dim = observation.shape
        assert state_dim == self.dim, (
            f"State dimension mismatch: expected {self.dim}, got {state_dim}"
        )
        
        # Flatten the batch and time dimensions for mapping
        flattened_obs = observation.view(batch_size * time_steps, state_dim)
        
        # Map the observations to their respective keys
        new_obs = {}
        for key, value in self.map.items():
            # Extract the indices for this key
            indices = value[0]
            new_obs[key] = flattened_obs[:, indices].view(batch_size, time_steps, -1)
        
        return new_obs

        
    def create_full_attention_mapping(self, env_name: str, dim: int):
        """
        Creates a mapping where the system attends to all observation and action indices.
        Args:
            dim: Total number of dimensions (e.g., 7, 14, 21, 28 for the Panda robot).

        Returns:
            A dictionary mapping observation and action spaces to all indices.
        """
        mapping = {
            env_name: torch.Tensor([list(range(dim)), list(range(dim))])
        }
        return mapping
    
    def create_shortest_path_matrix(dim):
        """
        Creates a shortest path matrix where the distance between indices i and j is |i - j|.
        Args:
            dim: Total number of dimensions (e.g., 7, 14, 21, 28).

        Returns:
            A torch.Tensor representing the shortest path matrix.
        """
        # Create a matrix where the entry (i, j) is |i - j|
        shortest_path_matrix = torch.abs(
            torch.arange(dim).unsqueeze(0) - torch.arange(dim).unsqueeze(1)
        )
        return shortest_path_matrix
    

# # Example usage
# dim_position = 7  # Joint position control
# dim_velocity = 14  # Velocity control
# dim_acceleration = 21  # Acceleration control
# dim_jerk = 28  # Jerk control

# position_mapping = create_full_attention_mapping(dim_position)
# velocity_mapping = create_full_attention_mapping(dim_velocity)
# acceleration_mapping = create_full_attention_mapping(dim_acceleration)
# jerk_mapping = create_full_attention_mapping(dim_jerk)

# print("Position Mapping:", position_mapping)
# print("Velocity Mapping:", velocity_mapping)
# print("Acceleration Mapping:", acceleration_mapping)
# print("Jerk Mapping:", jerk_mapping)
