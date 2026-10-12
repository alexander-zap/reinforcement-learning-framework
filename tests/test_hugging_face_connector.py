"""HuggingFaceConnector upload/download with the Hugging Face Hub mocked (no network).

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from rl_framework.util import (
    HuggingFaceConnector,
    HuggingFaceDownloadConfig,
    HuggingFaceUploadConfig,
)
from rl_framework.util.connector import hugging_face_connector


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Mocked Hub: the repo snapshot is a local folder; uploaded files and folders are recorded."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    api = Mock()
    api.create_repo.return_value = "https://huggingface.co/user/repo"
    uploaded_files = {}
    api.upload_file.side_effect = lambda path_or_fileobj, **kwargs: uploaded_files.update(
        {kwargs["path_in_repo"]: Path(path_or_fileobj).read_bytes()}
    )
    monkeypatch.setattr(hugging_face_connector, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(hugging_face_connector, "snapshot_download", Mock(return_value=str(snapshot)))
    recorded_videos = []
    monkeypatch.setattr(hugging_face_connector, "record_video", lambda **kwargs: recorded_videos.append(kwargs))
    return {"api": api, "snapshot": snapshot, "uploaded_files": uploaded_files, "videos": recorded_videos}


def make_connector(video_length=0):
    return HuggingFaceConnector(
        upload_config=HuggingFaceUploadConfig(
            upload=True,
            file_name="agent.zip",
            video_length=video_length,
            repository_id="user/repo",
            environment_name="CartPole-v1",
            model_architecture="PPO",
            commit_message="upload",
        ),
        download_config=HuggingFaceDownloadConfig(download=True, file_name="agent.zip", repository_id="user/repo"),
    )


def make_agent():
    agent = Mock()
    agent.save_to_file.side_effect = lambda path: Path(path).write_bytes(b"model")
    return agent


def test_checkpoint_upload_pushes_only_the_model_file(hub):
    make_connector().upload(agent=make_agent(), checkpoint_id=5)

    assert hub["uploaded_files"] == {"./checkpoints/checkpoint-5_agent.zip": b"model"}
    hub["api"].upload_folder.assert_not_called()


def test_final_upload_writes_model_results_logs_and_model_card(hub):
    connector = make_connector()
    connector.log_value(200.0, "mean_reward")
    connector.log_value(5.0, "std_reward")
    connector.log_value_with_timestep(10, 1.5, "Episode reward")
    connector.log_dict({"a": 1}, "settings")

    connector.upload(agent=make_agent())

    snapshot = hub["snapshot"]
    assert (snapshot / "agent.zip").read_bytes() == b"model"
    results = json.loads((snapshot / "results.json").read_text())
    assert results["env_id"] == "CartPole-v1"
    assert results["mean_reward"] == 200.0
    assert json.loads((snapshot / "logged_values.json").read_text()) == {"Episode reward": [[10, 1.5]]}
    assert json.loads((snapshot / "logged_dicts.json").read_text()) == {"settings": {"a": 1}}
    assert (snapshot / "system.json").exists()
    readme = (snapshot / "README.md").read_text(encoding="utf-8")
    assert "200.00 +/- 5.00" in readme
    assert "CartPole-v1" in readme
    hub["api"].upload_folder.assert_called_once()
    assert hub["api"].upload_folder.call_args.kwargs["commit_message"] == "upload"
    assert hub["videos"] == []


def test_final_upload_keeps_existing_readme_text(hub):
    (hub["snapshot"] / "README.md").write_text("My own model card", encoding="utf-8")

    make_connector().upload(agent=make_agent())

    readme = (hub["snapshot"] / "README.md").read_text(encoding="utf-8")
    assert "My own model card" in readme
    assert "not evaluated" in readme


def test_final_upload_records_video_into_the_repo(hub):
    environment = object()
    make_connector(video_length=10).upload(agent=make_agent(), video_recording_environment=environment)

    assert len(hub["videos"]) == 1
    assert hub["videos"][0]["video_recording_environment"] is environment
    assert hub["videos"][0]["file_path"] == hub["snapshot"] / "replay.mp4"


def test_final_upload_serializes_numpy_values_logged_during_training(hub):
    connector = make_connector()
    connector.log_histogram_with_timestep(10, np.array([0, 1, 1]), "Action distribution")
    connector.log_value_with_timestep(np.int64(10), np.float32(1.5), "Episode reward")

    connector.upload(agent=make_agent())

    assert json.loads((hub["snapshot"] / "logged_histograms.json").read_text()) == {
        "Action distribution": [[[0, 1, 1], 10]]
    }
    assert json.loads((hub["snapshot"] / "logged_values.json").read_text()) == {"Episode reward": [[10, 1.5]]}


def test_final_upload_still_rejects_values_which_are_not_json_serializable(hub):
    connector = make_connector()
    connector.log_dict({"callback": object()}, "settings")

    with pytest.raises(TypeError, match="not JSON serializable"):
        connector.upload(agent=make_agent())


def test_upload_requires_complete_upload_config(hub):
    connector = make_connector()
    connector.upload_config.commit_message = ""
    with pytest.raises(AssertionError):
        connector.upload(agent=make_agent())


def test_download_returns_the_hub_file_path(monkeypatch):
    hf_hub_download = Mock(return_value="/cache/agent.zip")
    monkeypatch.setattr(hugging_face_connector, "hf_hub_download", hf_hub_download)

    assert make_connector().download() == Path("/cache/agent.zip")
    hf_hub_download.assert_called_once_with("user/repo", "agent.zip")
