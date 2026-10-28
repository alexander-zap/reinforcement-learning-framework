import subprocess
import tempfile
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecVideoRecorder

from rl_framework.util.features_extractor_utils import (
    FeaturesExtractor,
    wrap_environment_with_features_extractor_preprocessor,
)


def record_video(
    agent, video_recording_environment, file_path: Path, fps: int = 1, video_length=1000, sb3_replay: bool = True
):
    """
    Generate a replay video of the agent.

    Args:
        agent (Agent): Agent to record video for.
        video_recording_environment (Environment): Environment used for clip creation.
        file_path (Path): Path where video should be saved to.
        fps (int): How many frame per seconds to record the video replay.
        video_length (int): How many time steps of the agent should be recorded
        sb3_replay (bool): Determines recording mode
            If True: Use SB3's VecVideoRecorder and FFMPEG to record video
            If False: Use simple recording method (saving RGB outputs from environment render)

    Like `Agent.evaluate`, the agent's `choose_action` gets the context of the episode (`reward` of its previous step,
    `episode_start` and `episode_key`), for agents whose actions depend on it (e.g., d3rlpy's decision transformers).

    The environment can be a gym.Env, a VecEnv (SB3 recording only; its first sub-environment is recorded) or an
    EnvironmentFactory (stub_environment, env_return_function): the environment is then created for the recording (of
    a list of environments, the first one) and closed afterwards.

    If the agent has a features extractor, the environment's observations are preprocessed with it (like in training
    and evaluation) before the agent chooses its actions; the recorded frames (`render`) are unaffected.

    Errors of the agent, the environment or the video conversion are raised (a failing ffmpeg as RuntimeError), and so
    are interrupts (KeyboardInterrupt): no video is saved then.
    """
    # tuple = EnvironmentFactory in format (stub_environment, env_return_function)
    environment_from_factory = isinstance(video_recording_environment, tuple)
    if environment_from_factory:
        _, env_func = video_recording_environment
        video_recording_environment = env_func()
        if isinstance(video_recording_environment, list):
            video_recording_environment, *unused_environments = video_recording_environment
            for unused_environment in unused_environments:
                unused_environment.close()

    try:
        _record_video(agent, video_recording_environment, file_path, fps, video_length, sb3_replay)
    finally:
        if environment_from_factory:
            # Closing twice is harmless (the SB3 recording closes its environment itself)
            video_recording_environment.close()


def _record_video(agent, video_recording_environment, file_path: Path, fps: int, video_length: int, sb3_replay: bool):
    features_extractor = getattr(agent, "features_extractor", None)
    if isinstance(features_extractor, FeaturesExtractor):
        video_recording_environment = wrap_environment_with_features_extractor_preprocessor(
            video_recording_environment, features_extractor
        )

    if sb3_replay:
        # Create a Stable-baselines3 vector environment (VecEnv)
        if not isinstance(video_recording_environment, VecEnv):
            if isinstance(video_recording_environment, gym.Env):
                video_recording_environment = DummyVecEnv([lambda: video_recording_environment])
            else:
                raise ValueError(
                    "For SB3 replay recording, the provided environment needs to be of type "
                    "gym.Env or stable_baselines3.common.vec_env.VecEnv."
                )

        # This is another temporary directory for video outputs (replay video file will be copied to repository path).
        # Copying is required, since SB3 creates other files which we don't want in the repo.
        with tempfile.TemporaryDirectory() as tmpdirname:
            env = VecVideoRecorder(
                video_recording_environment,
                tmpdirname,
                record_video_trigger=lambda x: x == 0,
                video_length=video_length,
                name_prefix="",
            )

            vectorized_observation = env.reset()
            reward, episode_start = 0.0, True

            try:
                # Note: Vectorized environments are automatically reset at the end of each episode.
                for _ in range(video_length):
                    action = agent.choose_action(
                        vectorized_observation[0],
                        reward=reward,
                        episode_start=episode_start,
                        episode_key=(id(env), 0),
                    )
                    vectorized_action = np.array([action])
                    vectorized_observation, rewards, dones, _ = env.step(vectorized_action)
                    # A done environment has started its next episode (automatic reset)
                    episode_start = bool(dones[0])
                    reward = 0.0 if episode_start else float(rewards[0])
            finally:
                # Save the video (also releases the video writer when recording fails or is interrupted)
                env.close()

            # Convert the video with x264 codec (argument list without a shell: paths may contain spaces)
            command = ["ffmpeg", "-y", "-i", str(env.video_path), "-vcodec", "h264", str(file_path)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {result.stderr}")
    else:
        images = []
        while len(images) < video_length:
            done = False
            observation, _ = video_recording_environment.reset()
            reward, episode_start = 0.0, True
            img = video_recording_environment.render()
            images.append(img)
            while not done:
                action = agent.choose_action(
                    observation,
                    reward=float(reward),
                    episode_start=episode_start,
                    episode_key=(id(video_recording_environment), 0),
                )
                episode_start = False
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = video_recording_environment.step(action)
                done = terminated or truncated
                img = video_recording_environment.render()
                images.append(img)
        imageio.mimsave(file_path, [np.array(img) for i, img in enumerate(images)], fps=fps)
