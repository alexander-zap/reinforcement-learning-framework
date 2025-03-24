import os
from pathlib import Path

import d3rlpy.algos
import gymnasium as gym
import imitation.algorithms.bc
import numpy as np
from clearml import Task
from imitation.data import rollout, serialize
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import PPO, MlpPolicy

from rl_framework.agent.imitation import D3RLPYAgent, EpisodeSequence, ImitationAgent
from rl_framework.util import (
    ClearMLConnector,
    ClearMLDownloadConfig,
    ClearMLUploadConfig,
)


def create_and_save_trajectories_dataset(env, timesteps, trajectories_dataset_path) -> None:
    def train_expert(venv):
        expert = PPO(policy=MlpPolicy, env=venv, device="cpu")
        expert.learn(100_000)
        return expert

    def download_expert_policy(venv):
        policy = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name="CartPole-v1",
            venv=venv,
        )
        policy.to("cpu")
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
TRAJECTORIES_PATH = (Path(__file__).parent.parent / "data" / "cartpole_rollout").as_posix()
VALIDATION_TRAJECTORIES_PATH = (Path(__file__).parent.parent / "data" / "cartpole_rollout").as_posix()

OFFLINE_RL = True
N_TRAINING_TIMESTEPS = 200000
N_EVALUATION_EPISODES = 10

if __name__ == "__main__":
    # Create environment(s); multiple environments for parallel training (used for hybrid IL approaches)
    environments = [gym.make("CartPole-v1", render_mode="rgb_array") for _ in range(PARALLEL_ENVIRONMENTS)]

    # Create connector
    task = Task.init(project_name="synthetic-player", auto_connect_frameworks={"pytorch": False})
    upload_connector_config = ClearMLUploadConfig(
        upload=True,
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
    if OFFLINE_RL:
        agent = D3RLPYAgent(algorithm_class=d3rlpy.algos.DQN, algorithm_parameters={"device": "cuda", "batch_size": 64})
    else:
        agent = ImitationAgent(algorithm_class=imitation.algorithms.bc.BC, algorithm_parameters={"minibatch_size": 16})

    if DOWNLOAD_EXISTING_AGENT:
        # Download existing agent from repository
        agent.download(connector=connector)

    if N_TRAINING_TIMESTEPS > 0:
        # Train agent
        if not os.path.exists(TRAJECTORIES_PATH):
            create_and_save_trajectories_dataset(environments[0], N_TRAINING_TIMESTEPS, TRAJECTORIES_PATH)

        if not os.path.exists(VALIDATION_TRAJECTORIES_PATH):
            create_and_save_trajectories_dataset(
                environments[0], N_TRAINING_TIMESTEPS * 0.1, VALIDATION_TRAJECTORIES_PATH
            )

        sequence = EpisodeSequence.from_dataset(TRAJECTORIES_PATH, loop=True)
        validation_sequence = EpisodeSequence.from_dataset(VALIDATION_TRAJECTORIES_PATH, loop=True)
        agent.train(
            episode_sequence=sequence,
            validation_episode_sequence=validation_sequence,
            training_environments=environments,
            total_timesteps=N_TRAINING_TIMESTEPS,
            connector=connector,
        )

    # Evaluate the model
    mean_reward, std_reward = agent.evaluate(
        evaluation_environment=environments[0], n_eval_episodes=N_EVALUATION_EPISODES
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    connector.log_value(mean_reward, "mean_reward")
    connector.log_value(std_reward, "std_reward")

    # Upload the model
    agent.upload(connector=connector, video_recording_environment=environments[0])
