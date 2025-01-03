import os

import gymnasium as gym
import numpy as np
from clearml import Task
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout, serialize
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import PPO, MlpPolicy

from rl_framework.agent.imitation import EpisodeSequence, ImitationAgent
from rl_framework.util import (
    ClearMLConnector,
    ClearMLDownloadConfig,
    ClearMLUploadConfig,
)


def create_and_save_trajectories_dataset(env, timesteps, trajectories_dataset_path) -> None:
    def train_expert(venv):
        expert = PPO(
            policy=MlpPolicy,
            env=env,
        )
        expert.learn(100_000)
        return expert

    def download_expert_policy(venv):
        policy = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name="CartPole-v1",
            venv=venv,
        )
        return policy

    environment_return_functions = [lambda: RolloutInfoWrapper(Monitor(env))]
    vectorized_environment = DummyVecEnv(env_fns=environment_return_functions)
    expert_policy = download_expert_policy(vectorized_environment)

    rollouts = rollout.rollout(
        expert_policy,
        vectorized_environment,
        rollout.make_sample_until(min_timesteps=timesteps),
        rng=np.random.default_rng(0),
    )

    serialize.save(trajectories_dataset_path, rollouts)


PARALLEL_ENVIRONMENTS = 8
DOWNLOAD_EXISTING_AGENT = False
TRAJECTORIES_PATH = "../data/test_rollouts"

N_TRAINING_TIMESTEPS = 200000
N_EVALUATION_EPISODES = 10

if __name__ == "__main__":
    # Create environment(s); multiple environments for parallel training (used for hybrid IL approaches)
    environments = [gym.make("CartPole-v1", render_mode="rgb_array") for _ in range(PARALLEL_ENVIRONMENTS)]

    # Create connector
    task = Task.init(project_name="synthetic-player", auto_connect_frameworks={"pytorch": False})
    upload_connector_config = ClearMLUploadConfig(
        file_name="agent.zip",
        video_length=0,
    )
    download_connector_config = ClearMLDownloadConfig(
        model_id="", file_name="agent.zip", download=DOWNLOAD_EXISTING_AGENT
    )
    connector = ClearMLConnector(
        task=task, upload_config=upload_connector_config, download_config=download_connector_config
    )

    # Create new agent
    agent = ImitationAgent(algorithm_class=GAIL, algorithm_parameters={})

    if DOWNLOAD_EXISTING_AGENT:
        # Download existing agent from repository
        agent.download(connector=connector)

    if N_TRAINING_TIMESTEPS > 0:
        # Train agent
        if not os.path.exists(TRAJECTORIES_PATH):
            create_and_save_trajectories_dataset(environments[0], N_TRAINING_TIMESTEPS, TRAJECTORIES_PATH)

        sequence = EpisodeSequence.from_dataset(TRAJECTORIES_PATH)
        agent.train(
            episode_sequence=sequence,
            training_environments=environments,
            total_timesteps=N_TRAINING_TIMESTEPS,
            connector=connector,
        )

    # Evaluate the model
    mean_reward, std_reward = agent.evaluate(
        evaluation_environment=environments[0], n_eval_episodes=N_EVALUATION_EPISODES
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Upload the model
    agent.upload(
        connector=connector,
        video_recording_environment=environments[0],
        variable_values_to_log={"mean_reward": mean_reward, "std_reward": std_reward},
    )
