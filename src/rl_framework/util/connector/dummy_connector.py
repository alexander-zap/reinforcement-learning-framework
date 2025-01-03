from pathlib import Path

from .base_connector import Connector, DownloadConfig, UploadConfig


class DummyConnector(Connector):
    def __init__(self, upload_config: UploadConfig = None, download_config: DownloadConfig = None):
        super().__init__(upload_config, download_config)

    def upload(
        self, agent, evaluation_environment, variable_values_to_log=None, checkpoint_id=None, *args, **kwargs
    ) -> None:
        pass

    def download(self, *args, **kwargs) -> Path:
        pass
