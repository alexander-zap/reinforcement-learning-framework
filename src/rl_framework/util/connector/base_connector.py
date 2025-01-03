from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, SupportsFloat, Text, Tuple


@dataclass
class UploadConfig(ABC):
    """
    file_name (Text): Name of the file the agent model should be saved to (uploaded model will be named accordingly).
    video_length (int): Length of video in frames (which should be generated and uploaded to the connector).
        No video is uploaded if length is 0 or negative.
    """

    file_name: str
    video_length: int

    def get_config_dict(self) -> Dict:
        return vars(self)


@dataclass
class DownloadConfig(ABC):
    """
    file_name (str): File name of previously saved agent model (the saved agent file previously uploaded).
    """

    download: bool
    file_name: str

    def get_config_dict(self) -> Dict:
        return vars(self)


class Connector(ABC):
    def __init__(self, upload_config: UploadConfig, download_config: DownloadConfig):
        """Initialize connector for uploading or downloading agents from the HuggingFace Hub.

        All attributes which are relevant for uploading or downloading are set in the config object.
        This is a conscious design decision to enable generic exchange of connectors and prevent the requirements
        of parameter passing for upload/download method calls.

        For repeated use of the same connector instance, change the attributes of the config object which is passed
        to the upload/download method.

        Args:
            upload_config: Connector configuration data for uploading a model to the connector.
            download_config: Connector configuration data for downloading a model from the connector.
        NOTE: See individual connector package for the documented config dataclass attributes.

        self attributes:
            logging_history: Dictionary mapping each logged value name to a list of logged values, e.g.:
            {
                "Episode reward": [(50.6, 10), (90.5, 20), (150.3, 30), (200.0, 40)],
                "Epsilon": [(1.0, 10), (0.74, 20), (0.46, 30), (0.15, 10)]
            }
            Elements of each list are tuples of timestep-value-points.
        """
        self.upload_config = upload_config
        self.download_config = download_config
        self.logging_history: Dict[Text, List[Tuple]] = defaultdict(list)

    def log_value(self, timestep: int, value_scalar: SupportsFloat, value_name: Text) -> None:
        """
        Log scalar value to create a sequence of values over time steps.
        Can be used afterward for visualization (e.g., plotting of value over time).

        Args:
            timestep: Time step which the scalar value corresponds to (x-value)
            value_scalar: Scalar value which should be logged (y-value)
            value_name: Name of scalar value (e.g., "avg. sum of reward")
        """
        self.logging_history[value_name].append((timestep, value_scalar))

    @abstractmethod
    def upload(
        self, agent, evaluation_environment, variable_values_to_log=None, checkpoint_id=None, *args, **kwargs
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def download(self, *args, **kwargs) -> Path:
        raise NotImplementedError
