import torch
DIMS = {
    'position': 7,
    'velocity': 14,
    'acceleration': 21,
    'jerk': 28
}

class Mapping:
    def __init__(self, env_name: str):
        self.dim = DIMS[env_name]
        self.map = self.create_full_attention_mapping(env_name, self.dim)
        
        self.shortest_path_matrix = self.create_shortest_path_matrix(self.dim)
        
    def get_map(self):
        return self.map

    def create_observation(self, observation: torch.Tensor):
        observation = observation.reshape(observation.shape[0], self.dim)
        new_obs = {}
        for key, value in self.map.items():
            new_obs[key] = observation[:, value[0]]
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
