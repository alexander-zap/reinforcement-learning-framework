import datetime
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Text

import stable_baselines3
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from rl_framework.util.types import Environment
from rl_framework.util.video_recording import record_video

from .base_connector import Connector, DownloadConfig, UploadConfig


@dataclass
class HuggingFaceUploadConfig(UploadConfig):
    """
    repository_id (Text): Repository ID from the HF Hub we want to upload to.
    environment_name (Text): Name of the environment (only used for the model card).
    model_architecture (Text): Name of the used model architecture (only used for model card and metadata).
    commit_message (Text): Commit message for the HuggingFace repository commit.
    n_eval_episodes (int): Number of episodes for agent evaluation to compute evaluation metrics
    video_length (int): Length of video in frames (which should be generated and uploaded to the connector).
        No video is uploaded if length is 0 or negative.
    """

    repository_id: Text
    environment_name: Text
    model_architecture: Text
    commit_message: Text


@dataclass
class HuggingFaceDownloadConfig(DownloadConfig):
    """
    repository_id (Text): Repository ID from the HF Hub of the RL agent we want to download.
    """

    repository_id: Text


class HuggingFaceConnector(Connector):
    def __init__(self, upload_config: HuggingFaceUploadConfig, download_config: HuggingFaceDownloadConfig):
        """
        Initialize the connector for HuggingFace.

        Args:
            upload_config (UploadConfig): Connector configuration data for uploading to HuggingFace.
            download_config (DownloadConfig): Connector configuration data for downloading from HuggingFace.
        NOTE: See above for the documented config dataclass attributes.
        """
        super().__init__(upload_config, download_config)

    def upload(
        self,
        agent,
        video_recording_environment: Optional[Environment] = None,
        checkpoint_id: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Evaluate, generate a video and upload a model to Hugging Face Hub.
        This method does the complete pipeline:
        - It evaluates the model
        - It generates the model card
        - It generates a replay video of the agent
        - It pushes everything to the Hub

        Args:
            agent (Agent): Agent (and its .algorithm attribute) to be uploaded.
            video_recording_environment (Environment): Environment used for clip creation before upload.
                If not provided, no video will be created.
            checkpoint_id (int): If specified, we do not perform a final upload with evaluating and generating but
                instead upload only a model checkpoint to a "checkpoints" folder.

        NOTE: If after running the package_to_hub function, and it gives an issue of rebasing, please run the
            following code: `cd <path_to_repo> && git add . && git commit -m "Add message" && git pull`
            And don't forget to do a `git push` at the end to push the change to the hub.
        """

        repository_id = self.upload_config.repository_id
        environment_name = self.upload_config.environment_name
        file_name = self.upload_config.file_name
        model_architecture = self.upload_config.model_architecture
        commit_message = self.upload_config.commit_message
        video_length = self.upload_config.video_length

        assert environment_name and file_name and model_architecture and commit_message

        _, repo_name = repository_id.split("/")

        api = HfApi()

        # TODO: Improve readability through sub-functions
        # Step 1: Create the repo
        repo_url = api.create_repo(
            repo_id=repository_id,
            exist_ok=True,
        )

        if checkpoint_id:
            # Step 2: Save the model checkpoint
            logging.debug(f"Pushing checkpoint #{checkpoint_id} to {repository_id} on the HuggingFace hub.")
            with tempfile.TemporaryDirectory("w") as tmp_dir:
                model_file_name = f"checkpoint-{checkpoint_id}_{file_name}"
                tmp_path = Path(tmp_dir) / model_file_name
                agent.save_to_file(tmp_path)
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    repo_id=repository_id,
                    path_in_repo=f"./checkpoints/{model_file_name}",
                    commit_message=f"Checkpoint upload #{checkpoint_id}",
                )
        else:
            logging.info(
                "This function will save your agent, evaluate its performance, generate a video of your agent, "
                "create a HuggingFace model card and push everything to the HuggingFace hub. "
            )

            # Step 2: Download files
            repo_local_path = Path(snapshot_download(repo_id=repository_id))

            # Step 3: Save the model
            agent.save_to_file(repo_local_path / file_name)

            # Step 4: update the JSON with everything stored in variable_values_to_log
            result_data = {
                "env_id": environment_name,
                "datetime": datetime.datetime.now().isoformat(),
            }
            for key, value in self.values_to_log.items():
                result_data[key] = value

            # Write a JSON file called "results.json" that will contain the evaluation results
            with open(repo_local_path / "results.json", "w") as outfile:
                json.dump(result_data, outfile)

            # Additionally write a JSON file for all manually logged sequences
            with open(repo_local_path / "logged_values.json", "w") as outfile:
                json.dump(self.value_sequences_to_log, outfile)
            with open(repo_local_path / "logged_histograms.json", "w") as outfile:
                json.dump(self.histogram_sequences_to_log, outfile)

            # Step 5: Create a system info file
            with open(repo_local_path / "system.json", "w") as outfile:
                env_info, _ = stable_baselines3.get_system_info()
                json.dump(env_info, outfile)

            # Step 6: Create the model card (README.md)

            agent_class_name = type(agent).__qualname__
            # FIXME: This is just a work-around.
            #  The real algorithm class (instead of `model_architecture`) and enum class name should be passed.
            algorithm_enum_class_name = type(agent).__qualname__.replace("Agent", "Algorithm")

            model_card = f"""

# Custom implemented {model_architecture} agent playing on *{environment_name}*

This is a trained model of an agent playing on the environment *{environment_name}*.
The agent was trained with a {model_architecture} algorithm.
See further agent and evaluation metadata in the according README section.


## Import
The Python module used for training and uploading/downloading is
[reinforcement-learning-framework](https://github.com/alexander-zap/reinforcement-learning-framework).
It is an easy-to-read, plug-and-use Reinforcement Learning framework and provides standardized interfaces
and implementations to various Reinforcement Learning methods and environments.

Also it provides connectors for the upload and download to popular model version control systems,
including the HuggingFace Hub.

## Usage
```python

from rl_framework import {agent_class_name}, {algorithm_enum_class_name}

# Create new agent instance
agent = {agent_class_name}(
    algorithm={algorithm_enum_class_name}.{model_architecture}
    algorithm_parameters={{
        ...
    }},
)

# Download existing agent from HF Hub
repository_id = "{repository_id}"
file_name = "{file_name}"
agent.download(repository_id=repository_id, filename=file_name)

```

Further examples can be found in the exploration section of the
[reinforcement-learning-framework repository](https://github.com/alexander-zap/reinforcement-learning-framework).

            """

            readme_path = repo_local_path / "README.md"
            if readme_path.exists():
                with readme_path.open("r", encoding="utf8") as f:
                    readme = f.read()
            else:
                readme = model_card

            with readme_path.open("w", encoding="utf-8") as f:
                f.write(readme)

            metadata = {
                "tags": [environment_name, "reinforcement-learning"],
            }

            metrics_value = "not evaluated"
            mean_reward = self.values_to_log.get("mean_reward")
            std_reward = self.values_to_log.get("std_reward")
            if (
                mean_reward is not None
                and std_reward is not None
                and isinstance(mean_reward, float)
                and isinstance(std_reward, float)
            ):
                metrics_value = f"{mean_reward:.2f} +/- {std_reward:.2f}"

            # Add mean_reward metric (use "not evaluated" value if not specified in variable_values_to_log)
            metadata_eval = metadata_eval_result(
                model_pretty_name=repo_name,
                task_pretty_name="reinforcement-learning",
                task_id="reinforcement-learning",
                metrics_pretty_name="mean_reward",
                metrics_id="mean_reward",
                metrics_value=metrics_value,
                dataset_pretty_name=environment_name,
                dataset_id=environment_name,
            )

            # Merges both dictionaries
            metadata = {**metadata, **metadata_eval}

            # Save our metrics to Readme metadata
            metadata_save(readme_path, metadata)

            # Step 6: Record a video
            if video_recording_environment and video_length > 0:
                video_path = repo_local_path / "replay.mp4"
                record_video(
                    agent=agent,
                    video_recording_environment=video_recording_environment,
                    file_path=video_path,
                    fps=1,
                    video_length=video_length,
                )

            # Step 7. Push everything to the Hub
            logging.info(f"Pushing repo {repository_id} to the Hugging Face Hub")
            api.upload_folder(
                repo_id=repository_id,
                folder_path=repo_local_path,
                path_in_repo=".",
                commit_message=commit_message,
            )

            logging.info(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")

    def download(self, *args, **kwargs) -> Path:
        """
        Download a reinforcement learning agent from the HuggingFace Hub.
        """
        repository_id = self.download_config.repository_id
        file_name = self.download_config.file_name

        assert repository_id and file_name

        file_path = hf_hub_download(repository_id, file_name)
        return Path(file_path)
