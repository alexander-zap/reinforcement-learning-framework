import logging
import os
import tempfile
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecVideoRecorder


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
    """

    if sb3_replay:
        # Create a Stable-baselines3 vector environment (VecEnv)
        if not isinstance(video_recording_environment, VecEnv):
            video_recording_environment = DummyVecEnv([lambda: video_recording_environment])

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

            try:
                # Note: Vectorized environments are automatically reset at the end of each episode.
                for _ in range(video_length):
                    action = agent.choose_action(vectorized_observation[0])
                    vectorized_action = np.array([action])
                    vectorized_observation, _, _, _ = env.step(vectorized_action)

                # Save the video
                env.close()

                # Convert the video with x264 codec
                inp = env.video_path
                out = file_path
                os.system(f"ffmpeg -y -i {inp} -vcodec h264 {out}")

            except KeyboardInterrupt:
                pass
            except Exception as e:
                logging.error(str(e))
    else:
        images = []
        while len(images) < video_length:
            done = False
            observation, _ = video_recording_environment.reset()
            img = video_recording_environment.render()
            images.append(img)
            while not done:
                action = agent.choose_action(observation)
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
