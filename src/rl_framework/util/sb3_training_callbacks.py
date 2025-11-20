from collections import Counter, deque
from typing import Any, List, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


def add_callbacks_to_callback(callbacks_to_add: CallbackList, callback_to_be_added_to: BaseCallback):
    if callback_to_be_added_to is None:
        callback_to_be_added_to = CallbackList([])
    elif not isinstance(callback_to_be_added_to, CallbackList):
        callback_to_be_added_to = CallbackList([callback_to_be_added_to])

    for callback in callbacks_to_add.callbacks:
        if callback not in callback_to_be_added_to.callbacks:
            callback_to_be_added_to.callbacks.append(callback)


class LoggingCallback(BaseCallback):
    """
    A custom callback that logs after every done episode:
        - histogram of performed actions per episode
        - episode rewards
        - mean of any tracked metric (provided as "step_metric_<name>" in the info dict at each step)
        - reason of episode end (provided as "episode_end_reason" in the info dict on terminated step)
    """

    def __init__(self, connector, log_action_distribution=False, verbose=0):
        """
        Args:
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.connector = connector
        self.log_action_distribution = log_action_distribution

        self.episode_reward: Union[np.ndarray | None] = None
        # Tracking actions per episode (one list per agent)
        # NOTE: Even though officially actions can be Any, for logging we assume they are int or float
        self.episode_actions: List[List[Any]] = []

        # Tracking stats across multiple episodes
        self.episode_end_reasons = deque(maxlen=500)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        if self.episode_reward is None:
            self.episode_reward = self.locals["rewards"]
        else:
            self.episode_reward += self.locals["rewards"]

        if self.log_action_distribution:
            if not self.episode_actions:
                self.episode_actions = [[action] for action in self.locals["actions"]]
            else:
                for agent_index, action in enumerate(self.locals["actions"]):
                    self.episode_actions[agent_index].append(action)

        done_indices = np.where(self.locals["dones"] == True)[0]
        if done_indices.size != 0:
            for done_index in done_indices:
                if not self.locals["infos"][done_index].get("discard", False):
                    self.connector.log_value_with_timestep(
                        self.num_timesteps, self.episode_reward[done_index], "Episode reward"
                    )

                if self.locals["infos"][done_index].get("episode_end_reason", None) is not None:
                    self.episode_end_reasons.append(self.locals["infos"][done_index]["episode_end_reason"])
                    counter = Counter(self.episode_end_reasons)
                    for reason, count in counter.items():
                        self.connector.log_value_with_timestep(
                            self.num_timesteps, count / len(self.episode_end_reasons), f"Episode end reason - {reason}"
                        )

                if self.log_action_distribution:
                    if self.episode_actions[done_index]:
                        if isinstance(self.episode_actions[done_index][0], np.ndarray):
                            for action_index, action_sequence in enumerate(zip(*self.episode_actions[done_index])):
                                self.connector.log_histogram_with_timestep(
                                    self.num_timesteps,
                                    np.array(action_sequence),
                                    f"Action distribution - action dim {action_index}",
                                )
                        else:
                            self.connector.log_histogram_with_timestep(
                                self.num_timesteps, np.array(self.episode_actions[done_index]), "Action distribution"
                            )

                self.episode_actions[done_index] = []
                self.episode_reward[done_index] = 0

        return True


class SavingCallback(BaseCallback):
    """
    A custom callback which uploads the agent to the connector after every `checkpoint_frequency` steps.
    """

    def __init__(self, agent, connector, checkpoint_frequency=50000, verbose=0):
        """
        Args:
            checkpoint_frequency: After how many steps a checkpoint should be saved to the connector.
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.agent = agent
        self.connector = connector
        self.checkpoint_frequency = checkpoint_frequency
        self.next_upload = checkpoint_frequency

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        if self.num_timesteps > self.next_upload:
            self.connector.upload(
                agent=self.agent,
                checkpoint_id=self.num_timesteps,
            )
            self.next_upload = self.num_timesteps + self.checkpoint_frequency

        return True
