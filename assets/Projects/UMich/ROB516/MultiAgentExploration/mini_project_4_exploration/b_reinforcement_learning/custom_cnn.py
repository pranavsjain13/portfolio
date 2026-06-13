"""
Custom CNN Feature Extractor for Small Observations

This module provides a custom CNN architecture designed for small (11x11) observations,
which is below the 36x36 minimum required by Stable-Baselines3's default CnnPolicy.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor designed for small spatial observations (11x11).
    
    The default NatureCNN in Stable-Baselines3 requires 36x36 minimum resolution.
    This custom extractor uses smaller kernels and fewer layers to handle 11x11 inputs.
    
    Architecture:
    - Conv2d: 3 channels -> 16 channels, 3x3 kernel, stride 1, padding 1
    - ReLU activation
    - Conv2d: 16 channels -> 32 channels, 3x3 kernel, stride 1, padding 1
    - ReLU activation
    - Flatten
    - Linear: flattened_size -> 128 features
    
    Args:
        observation_space: The observation space of the environment
        features_dim: Number of features extracted (default: 128)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Validate observation space
        n_input_channels = observation_space.shape[-1]  # Channels last: (H, W, C)
        
        # Define the CNN architecture for small inputs
        self.cnn = nn.Sequential(
            # First conv layer: 3 -> 16 channels, maintains spatial dimensions
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second conv layer: 16 -> 32 channels, maintains spatial dimensions
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Flatten the output
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create a sample tensor with channels-last format (B, H, W, C)
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            # Convert to channels-first for PyTorch (B, C, H, W)
            sample_input = sample_input.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_input).shape[1]

        # Linear layer to reduce to desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Input observations with channels-last format (B, H, W, C)
            
        Returns:
            Extracted features of shape (B, features_dim)
        """
        # Convert from channels-last (B, H, W, C) to channels-first (B, C, H, W)
        observations = observations.permute(0, 3, 1, 2)
        
        # Forward through CNN
        features = self.cnn(observations)
        
        # Forward through linear layer
        features = self.linear(features)
        
        return features


def create_policy_kwargs(features_dim: int = 128) -> dict:
    """
    Create policy_kwargs for PPO with custom CNN feature extractor.
    
    Args:
        features_dim: Number of features to extract (default: 128)
        
    Returns:
        Dictionary to pass to PPO as policy_kwargs parameter
        
    Example:
        >>> from stable_baselines3 import PPO
        >>> policy_kwargs = create_policy_kwargs(features_dim=128)
        >>> model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
    """
    return {
        "features_extractor_class": SmallCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "normalize_images": False,  # Our observations are already normalized to [0, 1]
    }
