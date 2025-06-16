from pathlib import Path
from typing import Optional

import gymnasium as gym

from .base_connector import Connector, DownloadConfig, UploadConfig


class DummyConnector(Connector):
    def __init__(self, upload_config: UploadConfig = None, download_config: DownloadConfig = None):
        super().__init__(upload_config, download_config)

    def upload(
        self,
        agent,
        video_recording_environment: Optional[gym.Env] = None,
        checkpoint_id: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        pass

    def download(self, *args, **kwargs) -> Path:
        pass
