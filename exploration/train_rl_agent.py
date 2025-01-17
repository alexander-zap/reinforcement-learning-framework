import gymnasium as gym
import stable_baselines3
from clearml import Task

from rl_framework.agent.reinforcement import CustomAgent, StableBaselinesAgent
from rl_framework.agent.reinforcement.custom_algorithms import QLearning
from rl_framework.util import (
    ClearMLConnector,
    ClearMLDownloadConfig,
    ClearMLUploadConfig,
)

# Flag whether to use StablesBaselinesAgent or CustomAgent for this example
USE_SB3 = True

PARALLEL_ENVIRONMENTS = 8
DOWNLOAD_EXISTING_AGENT = False

N_TRAINING_TIMESTEPS = 100000
N_EVALUATION_EPISODES = 100

if __name__ == "__main__":
    # Create environment(s); multiple environments for parallel training
    environments = [gym.make("Taxi-v3", render_mode="rgb_array") for _ in range(PARALLEL_ENVIRONMENTS)]

    # Create connector
    task = Task.init(project_name="synthetic-player")
    upload_connector_config = ClearMLUploadConfig(
        upload=True,
        file_name="agent.zip",
        video_length=0,
    )
    download_connector_config = ClearMLDownloadConfig(model_id="", file_name="", download=DOWNLOAD_EXISTING_AGENT)
    connector = ClearMLConnector(
        task=task, upload_config=upload_connector_config, download_config=download_connector_config
    )

    # Create new agent
    if USE_SB3:
        agent = StableBaselinesAgent(
            algorithm_class=stable_baselines3.PPO,
            algorithm_parameters={
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
                "batch_size": 64,
                "gamma": 0.999,
                "verbose": 1,
            },
        )
    else:
        agent = CustomAgent(
            algorithm_class=QLearning,
            algorithm_parameters={
                "alpha": 0.1,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_min": 0.05,
                "n_actions": environments[0].action_space.n,
                "n_observations": environments[0].observation_space.n,
                "randomize_q_table": False,
            },
        )

    if DOWNLOAD_EXISTING_AGENT:
        # Download existing agent from repository
        agent.download(connector=connector)

    # Train agent
    agent.train(training_environments=environments, total_timesteps=N_TRAINING_TIMESTEPS, connector=connector)

    # Evaluate the model
    mean_reward, std_reward = agent.evaluate(
        evaluation_environment=environments[0], n_eval_episodes=N_EVALUATION_EPISODES
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    connector.log_value(mean_reward, "mean_reward")
    connector.log_value(std_reward, "std_reward")

    # Upload the model
    agent.upload(connector=connector, video_recording_environment=environments[0])
