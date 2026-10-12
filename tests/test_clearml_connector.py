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


def make_agent():
    agent = Mock()
    agent.save_to_file.side_effect = lambda path: Path(path).write_bytes(b"model")
    agent.save_policy_as_onnx.side_effect = lambda path: Path(path).write_bytes(b"onnx")
    return agent


def upload_checkpoint(file_name):
    connector = make_connector(upload_file_name=file_name)
    connector.upload(agent=make_agent(), checkpoint_id=7)
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


def test_upload_adds_policy_onnx_as_artifact():
    connector = make_connector(upload_file_name="model.zip")
    connector.upload(agent=make_agent(), checkpoint_id=7)
    kwargs = connector.task.upload_artifact.call_args.kwargs
    assert kwargs["name"] == "model-7_ONNX"
    assert kwargs["artifact_object"].endswith("-model-7.onnx")


def test_upload_skips_onnx_for_agents_without_onnx_export():
    connector = make_connector(upload_file_name="model.zip")
    agent = make_agent()
    agent.save_policy_as_onnx.side_effect = NotImplementedError
    connector.upload(agent=agent, checkpoint_id=7)
    connector.task.update_output_model.assert_called_once()
    connector.task.upload_artifact.assert_not_called()


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


def test_connector_sets_task_name_and_tags():
    task = Mock()
    ClearMLConnector(
        upload_config=ClearMLUploadConfig(
            upload=True, file_name="model.zip", video_length=0, task_name="run", task_tags=["a", "b"]
        ),
        download_config=ClearMLDownloadConfig(download=False),
        task=task,
    )
    task.set_name.assert_called_once_with("run")
    task.add_tags.assert_called_once_with(["a", "b"])


def test_connector_keeps_task_name_when_none_is_configured():
    connector = make_connector()
    connector.task.set_name.assert_not_called()
    connector.task.add_tags.assert_called_once_with([])


def test_log_dict_connects_configuration_and_keeps_local_copy():
    connector = make_connector()
    connector.log_dict({"level": 3}, "settings")

    connector.task.connect_configuration.assert_called_once_with({"level": 3}, name="settings")
    assert connector.dicts_to_log == {"settings": {"level": 3}}


def test_log_value_with_timestep_reports_scalar_with_title_fallback():
    connector = make_connector()
    connector.log_value_with_timestep(5, 1.5, "reward")
    connector.log_value_with_timestep(6, 2, "mean", title_name="speed")

    report_scalar = connector.task.get_logger.return_value.report_scalar
    assert [c.kwargs for c in report_scalar.call_args_list] == [
        {"title": "reward", "series": "reward", "value": 1.5, "iteration": 5},
        {"title": "speed", "series": "mean", "value": 2.0, "iteration": 6},
    ]
    assert connector.value_sequences_to_log == {"reward": [(5, 1.5)], "mean": [(6, 2)]}


def test_log_value_reports_rounded_single_value():
    connector = make_connector()
    connector.log_value(1.23456, "mean_reward")

    connector.task.logger.report_single_value.assert_called_once_with("mean_reward", 1.23)
    assert connector.values_to_log == {"mean_reward": 1.23456}


def test_log_histogram_reports_latest_histogram_as_figure():
    connector = make_connector()
    connector.log_histogram_with_timestep(3, [0, 1, 1, 2], "actions")

    kwargs = connector.task.get_logger.return_value.report_matplotlib_figure.call_args.kwargs
    assert kwargs["title"] == "000000000000003 - actions"
    assert kwargs["series"] == "actions"
    assert kwargs["iteration"] == 3


@pytest.mark.xfail(
    strict=True,
    reason="clearml_connector.py:106-113 creates a new seaborn FacetGrid figure per histogram and never closes it, "
    "so open matplotlib figures pile up during training",
)
def test_log_histogram_does_not_leak_matplotlib_figures():
    import matplotlib.pyplot as plt

    connector = make_connector()
    open_before = len(plt.get_fignums())
    for timestep in range(5):
        connector.log_histogram_with_timestep(timestep, [0, 1, 1, 2], "actions")

    assert len(plt.get_fignums()) == open_before


def test_final_upload_tags_model_and_uploads_system_info_and_video(monkeypatch):
    recorded = []
    monkeypatch.setattr(clearml_connector, "record_video", lambda **kwargs: recorded.append(kwargs))
    connector = ClearMLConnector(
        upload_config=ClearMLUploadConfig(upload=True, file_name="model.zip", video_length=10, model_tags=["best"]),
        download_config=ClearMLDownloadConfig(download=False),
        task=Mock(),
    )
    agent = make_agent()
    environment = object()

    connector.upload(agent=agent, video_recording_environment=environment)

    kwargs = connector.task.update_output_model.call_args.kwargs
    assert kwargs["name"] == "model"
    assert kwargs["tags"] == ["final", "best"]
    artifact_names = [c.kwargs["name"] for c in connector.task.upload_artifact.call_args_list]
    assert artifact_names == ["model_ONNX", "system_info"]
    assert len(recorded) == 1
    assert recorded[0]["agent"] is agent
    assert recorded[0]["video_recording_environment"] is environment
    assert recorded[0]["video_length"] == 10
    connector.task.get_logger.return_value.report_media.assert_called_once()


def test_final_upload_without_video_environment_records_no_video(monkeypatch):
    recorded = []
    monkeypatch.setattr(clearml_connector, "record_video", lambda **kwargs: recorded.append(kwargs))
    connector = make_connector()

    connector.upload(agent=make_agent())

    assert recorded == []
    connector.task.get_logger.return_value.report_media.assert_not_called()


def test_checkpoint_upload_is_tagged_as_checkpoint():
    connector = make_connector()
    connector.upload(agent=make_agent(), checkpoint_id=7)
    assert connector.task.update_output_model.call_args.kwargs["tags"] == ["checkpoint"]


def mock_input_model(monkeypatch, local_copy):
    input_model = Mock(return_value=Mock(get_local_copy=Mock(return_value=str(local_copy))))
    monkeypatch.setattr(clearml_connector, "InputModel", input_model)
    return input_model


def test_download_by_model_id_returns_local_copy(monkeypatch):
    input_model = mock_input_model(monkeypatch, __file__)
    connector = ClearMLConnector(
        upload_config=ClearMLUploadConfig(upload=False, file_name="model.zip", video_length=0),
        download_config=ClearMLDownloadConfig(download=True, model_id="model-id"),
        task=Mock(),
    )

    assert connector.download() == Path(__file__)
    input_model.assert_called_once_with("model-id")
    input_model.return_value.connect.assert_called_once()


def test_download_by_task_without_file_name_takes_last_output_model(monkeypatch):
    container = Mock()
    container.models = {"output": [Mock(id="old"), Mock(id="latest")]}
    monkeypatch.setattr(clearml_connector.Task, "get_task", Mock(return_value=container))
    input_model = mock_input_model(monkeypatch, __file__)

    make_connector(task_id="task-id").download()

    input_model.assert_called_once_with("latest")


def test_download_of_legacy_folder_model_returns_file_inside_folder(monkeypatch, tmp_path):
    mock_input_model(monkeypatch, tmp_path)
    connector = ClearMLConnector(
        upload_config=ClearMLUploadConfig(upload=False, file_name="model.zip", video_length=0),
        download_config=ClearMLDownloadConfig(download=True, file_name="model.zip", model_id="model-id"),
        task=Mock(),
    )

    assert connector.download() == tmp_path / "model.zip"


def test_download_without_model_or_task_id_fails():
    with pytest.raises(AssertionError, match="Neither model_id nor task_id"):
        make_connector().download()
