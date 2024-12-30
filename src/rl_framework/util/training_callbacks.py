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
    A custom callback that logs episode rewards after every done episode.
    """

    def __init__(self, connector, verbose=0):
        """
        Args:
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.connector = connector
        self.episode_reward = None

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        if self.episode_reward is None:
            self.episode_reward = self.locals["rewards"]
        else:
            self.episode_reward += self.locals["rewards"]

        done_indices = np.where(self.locals["dones"] == True)[0]
        if done_indices.size != 0:
            for done_index in done_indices:
                if not self.locals["infos"][done_index].get("discard", False):
                    self.connector.log_value(self.num_timesteps, self.episode_reward[done_index], "Episode reward")
                    self.episode_reward[done_index] = 0

        return True


class SavingCallback(BaseCallback):
    """
    A custom callback which uploads the agent to the connector after every `checkpoint_frequency` steps.
    """

    def __init__(self, agent, connector, evaluation_environment, checkpoint_frequency=50000, verbose=0):
        """
        Args:
            checkpoint_frequency: After how many steps a checkpoint should be saved to the connector.
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.agent = agent
        self.connector = connector
        self.evaluation_environment = evaluation_environment
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
                evaluation_environment=self.evaluation_environment,
                checkpoint_id=self.num_timesteps,
            )
            self.next_upload = self.num_timesteps + self.checkpoint_frequency

        return True
