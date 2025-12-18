from pathlib import Path
from typing import Optional

from rl_framework.util.connector import Connector, DownloadConfig, UploadConfig
from rl_framework.util.types import Environment


class DummyConnector(Connector):
    def __init__(self, upload_config: UploadConfig = None, download_config: DownloadConfig = None):
        super().__init__(upload_config, download_config)

    def upload(
        self,
        agent,
        video_recording_environment: Optional[Environment] = None,
        checkpoint_id: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        pass

    def download(self, *args, **kwargs) -> Path:
        pass
