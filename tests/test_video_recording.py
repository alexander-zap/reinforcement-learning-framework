"""record_video: simple frame recording and SB3 VecVideoRecorder + ffmpeg recording.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import shutil

import imageio
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.util import FeaturesExtractor, record_video, video_recording
from tests.toys import ConstantActionAgent, Toy, ToyParallel

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg executable")


def test_simple_recording_saves_whole_episodes_until_video_length_is_reached(tmp_path):
    file_path = tmp_path / "replay.gif"
    agent = ConstantActionAgent()

    record_video(agent, Toy(episode_length=3), file_path, video_length=5, sb3_replay=False)

    # Two episodes of (reset frame + 3 step frames).
    assert len(imageio.mimread(file_path)) == 8
    assert agent.observations


class Doubling(FeaturesExtractor):
    output_dim = 3

    def preprocess(self, observations):
        return np.asarray(observations) * 2

    def forward(self, observations):
        return observations


@pytest.mark.parametrize("sb3_replay", [False, True], ids=["simple", "sb3"])
def test_recording_preprocesses_observations_with_the_agents_features_extractor(tmp_path, monkeypatch, sb3_replay):
    monkeypatch.setattr(video_recording.os, "system", lambda command: None)
    agent = ConstantActionAgent(features_extractor=Doubling())
    environment = Toy(observation_space=spaces.Box(1, 1, (3,), np.float32), episode_length=3)

    record_video(agent, environment, tmp_path / "replay.gif", video_length=5, sb3_replay=sb3_replay)

    assert agent.observations
    assert all(np.array_equal(observation, [2, 2, 2]) for observation in agent.observations)


def test_recording_an_agent_with_features_extractor_requires_a_gym_environment(tmp_path):
    agent = ConstantActionAgent(features_extractor=Doubling())

    with pytest.raises(ValueError, match="features extractor requires a gym.Env"):
        record_video(agent, DummyVecEnv([Toy]), tmp_path / "replay.mp4", video_length=5)


def test_sb3_recording_rejects_unsupported_environments(tmp_path):
    with pytest.raises(ValueError, match="gym.Env or stable_baselines3"):
        record_video(ConstantActionAgent(), ToyParallel(), tmp_path / "replay.mp4")


@pytest.mark.parametrize("environment", [Toy(episode_length=3), DummyVecEnv([lambda: Toy(episode_length=3)])])
def test_sb3_recording_converts_the_recorded_video_with_ffmpeg(tmp_path, monkeypatch, environment):
    commands = []
    monkeypatch.setattr(video_recording.os, "system", commands.append)
    file_path = tmp_path / "replay.mp4"

    record_video(ConstantActionAgent(), environment, file_path, video_length=5)

    assert len(commands) == 1
    assert commands[0].startswith("ffmpeg -y -i ")
    assert commands[0].endswith(f"-vcodec h264 {file_path}")


def test_sb3_recording_logs_instead_of_raising_agent_errors(tmp_path, caplog):
    class FailingAgent(ConstantActionAgent):
        def choose_action(self, observation, deterministic=False, *args, **kwargs):
            raise RuntimeError("policy failed")

    record_video(FailingAgent(), Toy(), tmp_path / "replay.mp4", video_length=5)

    assert "policy failed" in caplog.text
    assert not (tmp_path / "replay.mp4").exists()


@needs_ffmpeg
def test_sb3_recording_writes_the_video(tmp_path):
    file_path = tmp_path / "replay.mp4"
    record_video(ConstantActionAgent(), Toy(episode_length=3), file_path, video_length=5)
    assert file_path.stat().st_size > 0


@needs_ffmpeg
@pytest.mark.xfail(
    strict=True,
    reason="video_recording.py:66 builds the ffmpeg command line without quoting, so paths with spaces break, "
    "and video_recording.py:70-71 only logs the failure",
)
def test_sb3_recording_writes_the_video_to_a_path_with_spaces(tmp_path):
    folder = tmp_path / "my videos"
    folder.mkdir()
    file_path = folder / "replay video.mp4"

    record_video(ConstantActionAgent(), Toy(episode_length=3), file_path, video_length=5)

    assert file_path.exists()
