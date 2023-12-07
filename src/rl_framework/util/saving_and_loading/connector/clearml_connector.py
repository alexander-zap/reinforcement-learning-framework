import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import stable_baselines3
from clearml import Task

from rl_framework.util import evaluate_agent
from rl_framework.util.video_recording import record_video

from .base_connector import Connector, DownloadConfig, UploadConfig


@dataclass
class ClearMLUploadConfig(UploadConfig):
    """
    n_eval_episodes (int): Number of episodes for agent evaluation to compute evaluation metrics
    """

    n_eval_episodes: int


@dataclass
class ClearMLDownloadConfig(DownloadConfig):
    """
    task_id (str): Id of the existing ClearML task to download the agent from
    """

    task_id: str


class ClearMLConnector(Connector):
    def __init__(self, task: Task):
        """
        Initialize the connector and pass a ClearML Task object for tracking parameters/artifacts/results.

        Args:
            task (Task): Active task object to track parameters/artifacts/results in the experiment run(s).
                See https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/ on how to use tasks for your purposes.
        """
        self.task = task

    def upload(self, connector_config: ClearMLUploadConfig, agent, evaluation_environment, *args, **kwargs) -> None:
        """Evaluate the agent on the evaluation environment and generate a video.
         Afterward, upload the artifacts and the agent itself to a ClearML task.

        Args:
            connector_config: Connector configuration data for uploading to HuggingFace.
                See above for the documented dataclass attributes.
            agent (Agent): Agent (and its .algorithm attribute) to be uploaded.
            evaluation_environment (Environment): Environment used for final evaluation and clip creation before upload.
        """
        logging.info(
            "This function will evaluate the performance of your agent and log the model as well as the experiment "
            "results as artifacts to ClearML. Also, a video of the agent's performance on the evaluation environment "
            "will be generated and uploaded to the 'Debug Sample' section of the ClearML experiment."
        )

        # Step 1: Save agent to temporary path and upload .zip file to ClearML
        with tempfile.TemporaryDirectory() as temp_path:
            logging.debug(f"Saving agent to .zip file at {temp_path} and uploading artifact ...")
            # TODO: This only works for SB3
            agent_save_path = Path(os.path.join(temp_path, "agent.zip"))
            agent.save_to_file(agent_save_path)
            while not os.path.exists(agent_save_path):
                time.sleep(1)
            self.task.upload_artifact(name="agent", artifact_object=temp_path)

        # Step 2: Evaluate the agent and upload a dictionary with evaluation metrics
        logging.debug("Evaluating agent and uploading experiment results ...")
        mean_reward, std_reward = evaluate_agent(
            agent=agent,
            evaluation_environment=evaluation_environment,
            n_eval_episodes=100,
        )
        experiment_result = {
            "mean_reward": round(mean_reward, 2),
            "std_reward": round(std_reward, 2),
        }
        self.task.upload_artifact(name="experiment_result", artifact_object=experiment_result)

        # Step 3: Create a system info dictionary and upload it
        logging.debug("Uploading system meta information ...")
        system_info, _ = stable_baselines3.get_system_info()
        self.task.upload_artifact(name="system_info", artifact_object=system_info)

        # Step 4: Record a video and log local video file
        temp_path = tempfile.mkdtemp()
        logging.debug(f"Recording video to {temp_path} and uploading as debug sample ...")
        video_path = Path(temp_path) / "replay.mp4"
        record_video(
            agent=agent,
            evaluation_environment=evaluation_environment,
            file_path=video_path,
            fps=1,
            video_length=1000,
            sb3_replay=False,
        )
        self.task.get_logger().report_media(
            "video ", "agent-in-environment recording", iteration=1, local_path=video_path
        )

        # TODO: Save README.md

    # TODO Implement downloading logic
    def download(self, connector_config: ClearMLDownloadConfig, *args, **kwargs) -> Path:
        raise NotImplementedError
