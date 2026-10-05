"""ClearMLConnector model file naming on upload and task-based download (ClearML mocked, no server).

Strict xfails document unresolved findings; they assert the desired behavior.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from rl_framework.util.connector import clearml_connector
from rl_framework.util.connector.clearml_connector import (
    ClearMLConnector,
    ClearMLDownloadConfig,
    ClearMLUploadConfig,
)


def make_connector(upload_file_name="model.zip", download_file_name=None, task_id=None):
    return ClearMLConnector(
        upload_config=ClearMLUploadConfig(upload=True, file_name=upload_file_name, video_length=0),
        download_config=ClearMLDownloadConfig(download=True, file_name=download_file_name, task_id=task_id),
        task=Mock(),
    )


def upload_checkpoint(file_name):
    connector = make_connector(upload_file_name=file_name)
    agent = Mock()
    agent.save_to_file.side_effect = lambda path: Path(path).write_bytes(b"model")
    connector.upload(agent=agent, checkpoint_id=7)
    kwargs = connector.task.update_output_model.call_args.kwargs
    return kwargs["name"], Path(kwargs["model_path"]).name


def test_upload_names_checkpoint_after_file_name():
    name, saved_file = upload_checkpoint("model.zip")
    assert name == "model-7"
    assert saved_file.endswith("-model-7.zip")


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason='clearml_connector.py:170 unpacks str.split(file_name, ".") into two names, failing for multi-dot names',
)
def test_upload_keeps_multi_dot_file_name():
    name, saved_file = upload_checkpoint("model.v2.zip")
    assert name == "model.v2-7"
    assert saved_file.endswith("-model.v2-7.zip")


def download_from_task(monkeypatch, file_name):
    container = Mock()
    container.models = {"output": {"model": Mock(id="model-id")}}
    monkeypatch.setattr(clearml_connector.Task, "get_task", Mock(return_value=container))
    input_model = Mock(return_value=Mock(get_local_copy=Mock(return_value=__file__)))
    monkeypatch.setattr(clearml_connector, "InputModel", input_model)

    make_connector(download_file_name=file_name, task_id="task-id").download()
    return input_model.call_args.args[0]


def test_download_by_task_finds_model_named_after_zip_file(monkeypatch):
    assert download_from_task(monkeypatch, "model.zip") == "model-id"


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="clearml_connector.py:228 strips file_name[:-4], assuming a 4-character extension like '.zip'",
)
def test_download_by_task_finds_model_named_after_other_extension(monkeypatch):
    assert download_from_task(monkeypatch, "model.pt") == "model-id"
