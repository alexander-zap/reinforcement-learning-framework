"""Connector base class bookkeeping, config dataclasses, DummyConnector, and Agent.upload/download delegation."""

from pathlib import Path

import numpy as np
import pytest

from rl_framework.util import Connector, DownloadConfig, DummyConnector, UploadConfig
from tests.toys import ConstantActionAgent, RecordingConnector


def test_connector_is_abstract():
    with pytest.raises(TypeError):
        Connector(upload_config=None, download_config=None)


def test_value_sequences_are_logged_per_name_as_timestep_value_pairs():
    connector = DummyConnector()
    connector.log_value_with_timestep(10, 1.5, "reward")
    connector.log_value_with_timestep(20, 2.5, "reward", title_name="ignored by the base class")
    connector.log_value_with_timestep(10, 0.1, "epsilon")

    assert connector.value_sequences_to_log == {"reward": [(10, 1.5), (20, 2.5)], "epsilon": [(10, 0.1)]}


def test_histograms_are_logged_per_name_as_value_timestep_pairs():
    connector = DummyConnector()
    connector.log_histogram_with_timestep(5, [0, 1, 1], "actions")
    connector.log_histogram_with_timestep(7, [1], "actions")

    assert connector.histogram_sequences_to_log == {"actions": [([0, 1, 1], 5), ([1], 7)]}


def test_single_values_are_stored_as_float_and_overwritten():
    connector = DummyConnector()
    connector.log_value(np.float32(1.25), "mean_reward")
    connector.log_value(3, "mean_reward")

    assert connector.values_to_log == {"mean_reward": 3.0}
    assert not isinstance(connector.values_to_log["mean_reward"], np.generic)


def test_dicts_are_stored_by_name():
    connector = DummyConnector()
    connector.log_dict({"a": 1}, "settings")
    connector.log_dict({"a": 2}, "settings")

    assert connector.dicts_to_log == {"settings": {"a": 2}}


def test_connectors_do_not_share_logged_values():
    first, second = DummyConnector(), DummyConnector()
    first.log_value_with_timestep(1, 1.0, "reward")

    assert second.value_sequences_to_log == {}


def test_dummy_connector_upload_and_download_are_no_ops():
    connector = DummyConnector()
    assert connector.upload(agent=object(), checkpoint_id=3) is None
    assert connector.download() is None


def test_config_dicts_contain_all_fields():
    upload_config = UploadConfig(upload=True, file_name="agent.zip", video_length=10)
    download_config = DownloadConfig(download=False, file_name="agent.zip")

    assert upload_config.get_config_dict() == {"upload": True, "file_name": "agent.zip", "video_length": 10}
    assert download_config.get_config_dict() == {"download": False, "file_name": "agent.zip"}


def test_agent_upload_delegates_to_connector():
    agent = ConstantActionAgent()
    connector = RecordingConnector()
    environment = object()

    agent.upload(connector, video_recording_environment=environment)

    assert connector.uploads == [{"agent": agent, "video_recording_environment": environment, "checkpoint_id": None}]


def test_agent_download_loads_the_downloaded_file_with_the_given_parameters():
    loaded = []

    class LoadingAgent(ConstantActionAgent):
        def load_from_file(self, file_path, algorithm_parameters, *args, **kwargs):
            loaded.append((file_path, algorithm_parameters))

    LoadingAgent().download(RecordingConnector(download_path=Path("agent.zip")), algorithm_parameters={"gamma": 0.9})

    assert loaded == [(Path("agent.zip"), {"gamma": 0.9})]
