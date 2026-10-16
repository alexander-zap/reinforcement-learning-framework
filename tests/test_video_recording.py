"""record_video: simple frame recording and SB3 VecVideoRecorder + ffmpeg recording."""

import shutil
from types import SimpleNamespace

import imageio
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.util import FeaturesExtractor, record_video, video_recording
from tests.toys import ConstantActionAgent, Toy, ToyParallel

FFMPEG_SUCCESS = SimpleNamespace(returncode=0, stderr="")
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
    monkeypatch.setattr(video_recording.subprocess, "run", lambda command, **kwargs: FFMPEG_SUCCESS)
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

    def run(command, **kwargs):
        commands.append(command)
        return FFMPEG_SUCCESS

    monkeypatch.setattr(video_recording.subprocess, "run", run)
    file_path = tmp_path / "replay.mp4"

    record_video(ConstantActionAgent(), environment, file_path, video_length=5)

    assert len(commands) == 1
    assert commands[0][:3] == ["ffmpeg", "-y", "-i"]
    assert commands[0][4:] == ["-vcodec", "h264", str(file_path)]


def test_sb3_recording_raises_ffmpeg_failures_with_their_output(tmp_path, monkeypatch):
    failure = SimpleNamespace(returncode=1, stderr="Invalid data found when processing input")
    monkeypatch.setattr(video_recording.subprocess, "run", lambda command, **kwargs: failure)

    with pytest.raises(RuntimeError, match="ffmpeg failed with exit code 1: Invalid data found when processing input"):
        record_video(ConstantActionAgent(), Toy(episode_length=3), tmp_path / "replay.mp4", video_length=5)


def test_sb3_recording_raises_agent_errors(tmp_path):
    class FailingAgent(ConstantActionAgent):
        def choose_action(self, observation, deterministic=False, *args, **kwargs):
            raise RuntimeError("policy failed")

    with pytest.raises(RuntimeError, match="policy failed"):
        record_video(FailingAgent(), Toy(), tmp_path / "replay.mp4", video_length=5)
    assert not (tmp_path / "replay.mp4").exists()


def test_interrupted_sb3_recording_stops_closes_the_recorder_and_saves_no_video(tmp_path, monkeypatch):
    class InterruptingAgent(ConstantActionAgent):
        def choose_action(self, observation, deterministic=False, *args, **kwargs):
            if len(self.observations) == 3:
                raise KeyboardInterrupt
            return super().choose_action(observation, deterministic, *args, **kwargs)

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return FFMPEG_SUCCESS

    monkeypatch.setattr(video_recording.subprocess, "run", run)
    agent = InterruptingAgent()
    environment = Toy(episode_length=10)

    with pytest.raises(KeyboardInterrupt):
        record_video(agent, environment, tmp_path / "replay.mp4", video_length=5)

    assert len(agent.observations) == 3
    assert commands == []
    assert environment.closed  # closing the recorder closes the environment it records
    assert not (tmp_path / "replay.mp4").exists()


@needs_ffmpeg
def test_sb3_recording_writes_the_video(tmp_path):
    file_path = tmp_path / "replay.mp4"
    record_video(ConstantActionAgent(), Toy(episode_length=3), file_path, video_length=5)
    assert file_path.stat().st_size > 0


@needs_ffmpeg
def test_sb3_recording_writes_the_video_to_a_path_with_spaces(tmp_path):
    folder = tmp_path / "my videos"
    folder.mkdir()
    file_path = folder / "replay video.mp4"

    record_video(ConstantActionAgent(), Toy(episode_length=3), file_path, video_length=5)

    assert file_path.exists()


class ContextRecordingAgent(ConstantActionAgent):
    """Records the episode context which `record_video` passes to `choose_action`."""

    def __init__(self):
        super().__init__()
        self.contexts = []
        self.episode_keys = set()

    def choose_action(self, observation, deterministic=False, *args, **kwargs):
        self.contexts.append((kwargs["reward"], kwargs["episode_start"]))
        self.episode_keys.add(kwargs["episode_key"])
        return super().choose_action(observation, deterministic, *args, **kwargs)


@pytest.mark.parametrize("sb3_replay", [False, True], ids=["simple", "sb3"])
def test_recording_passes_the_episode_context_to_the_agent(tmp_path, monkeypatch, sb3_replay):
    monkeypatch.setattr(video_recording.subprocess, "run", lambda command, **kwargs: FFMPEG_SUCCESS)
    agent = ContextRecordingAgent()

    record_video(agent, Toy(episode_length=3), tmp_path / "replay.gif", video_length=5, sb3_replay=sb3_replay)

    # Episodes of 3 steps with reward 1: the first step without reward, then the previous step's reward
    episode = [(0.0, True), (1.0, False), (1.0, False)]
    # Simple: whole episodes until 5 frames (4 per episode); SB3: 5 steps
    assert agent.contexts == (episode * 2 if not sb3_replay else (episode * 2)[:5])
    assert len(agent.episode_keys) == 1
