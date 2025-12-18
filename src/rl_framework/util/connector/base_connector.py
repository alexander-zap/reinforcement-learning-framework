from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, SupportsFloat, Text, Tuple

from rl_framework.util import Environment


@dataclass
class UploadConfig(ABC):
    """
    upload (bool): Flag whether an agent should be uploaded after training.
    file_name (Text): Name of the file the agent model should be saved to (uploaded model will be named accordingly).
    video_length (int): Length of video in frames (which should be generated and uploaded to the connector).
        No video is uploaded if length is 0 or negative.
    """

    upload: bool
    file_name: str
    video_length: int

    def get_config_dict(self) -> Dict:
        return vars(self)


@dataclass
class DownloadConfig(ABC):
    """
    download (bool): Flag whether an agent should be downloaded at the very beginning.
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
            histogram_sequences_to_log: Dictionary mapping histogram names to a list of logged histogram values, e.g.:
            {
                "action_distribution": [([0.1, 0.3, 0.6], 10), ([0.2, 0.5, 0.3], 20)]
            }
            Elements of each list are tuples of histogram-timestep-points.

            value_sequences_to_log: Dictionary mapping value names to a list of logged values, e.g.:
            {
                "Episode reward": [(50.6, 10), (90.5, 20), (150.3, 30), (200.0, 40)],
                "Epsilon": [(1.0, 10), (0.74, 20), (0.46, 30), (0.15, 10)]
            }
            Elements of each list are tuples of value-timestep-points.

            values_to_log: Dictionary mapping each logged value name to a single logged value, e.g.:
            {
                "Mean episode reward": 150.3,
                "Max episode reward": 200.0
            }
        """
        self.upload_config = upload_config
        self.download_config = download_config
        self.histogram_sequences_to_log: Dict[Text, List[Tuple]] = defaultdict(list)
        self.value_sequences_to_log: Dict[Text, List[Tuple]] = defaultdict(list)
        self.values_to_log: Dict[Text, SupportsFloat] = {}

    def log_histogram_with_timestep(
        self, timestep: int, histogram_values: List[SupportsFloat], histogram_name: Text
    ) -> None:
        """
        Log histogram values to create a sequence of histograms over time steps.
        Can be used afterward for visualization (e.g., plotting of histogram over time).

        Args:
            timestep: Time step which the histogram corresponds to
            histogram_values: Values which should be logged as histogram
            histogram_name: Name of histogram (e.g., "action_distribution")
        """
        self.histogram_sequences_to_log[histogram_name].append((histogram_values, timestep))

    def log_value_with_timestep(self, timestep: int, value_scalar: SupportsFloat, value_name: Text) -> None:
        """
        Log scalar value to create a sequence of values over time steps.
        Can be used afterward for visualization (e.g., plotting of value over time).

        Args:
            timestep: Time step which the scalar value corresponds to (x-value)
            value_scalar: Scalar value which should be logged (y-value)
            value_name: Name of scalar value (e.g., "episode_reward")
        """
        self.value_sequences_to_log[value_name].append((timestep, value_scalar))

    def log_value(self, metric_scalar: SupportsFloat, metric_name: Text) -> None:
        """
        Log one value with a certain metric name (once).

        Args:
            metric_scalar: Scalar value which should be logged
            metric_name: Name of scalar value (e.g., "mean_episode_reward")
        """
        self.values_to_log[metric_name] = float(metric_scalar)

    @abstractmethod
    def upload(
        self,
        agent,
        video_recording_environment: Optional[Environment] = None,
        checkpoint_id: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def download(self, *args, **kwargs) -> Path:
        raise NotImplementedError
