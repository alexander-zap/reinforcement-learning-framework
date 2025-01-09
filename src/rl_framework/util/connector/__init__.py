from .base_connector import Connector, DownloadConfig, UploadConfig
from .clearml_connector import (
    ClearMLConnector,
    ClearMLDownloadConfig,
    ClearMLUploadConfig,
)
from .dummy_connector import DummyConnector
from .hugging_face_connector import (
    HuggingFaceConnector,
    HuggingFaceDownloadConfig,
    HuggingFaceUploadConfig,
)
