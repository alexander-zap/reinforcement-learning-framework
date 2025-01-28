from abc import ABC, abstractmethod
from typing import Optional

import torch.nn


class CustomAlgorithm(ABC):
    @abstractmethod
    def __init__(self, features_extractor: Optional[torch.nn.Module], *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, connector, training_environments, total_timesteps, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation, deterministic, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, file_path, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, file_path, algorithm_parameters, *args, **kwargs):
        raise NotImplementedError
