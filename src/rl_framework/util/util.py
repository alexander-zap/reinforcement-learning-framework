from typing import Optional, List
import numpy as np
from tqdm import tqdm


def evaluate_agent(agent, evaluation_environment, n_eval_episodes: int,
                   seeds: Optional[List[int]] = None):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    Args:
        agent (Agent): Agent to evaluate
        evaluation_environment (Environment): The evaluation environment.
        n_eval_episodes (int): Number of episode to evaluate the agent.
        seeds (Optional[List[int]]): List of seeds for evaluations.
            No seed is used if not provided or fewer seeds are provided then n_eval_episodes.
    """
    if seeds is None:
        seeds = []
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        seed = seeds[episode] if episode < len(seeds) else None
        episode_reward = 0
        prev_observation = evaluation_environment.reset(seed=seed)
        prev_action = agent.choose_action(prev_observation, greedy=True)

        while True:
            observation, reward, done, info = evaluation_environment.step(prev_action)
            # next action to be executed (based on new observation)
            action = agent.choose_action(observation, greedy=True)
            episode_reward += reward
            prev_action = action

            if done:
                episode_rewards.append(episode_reward)
                break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
