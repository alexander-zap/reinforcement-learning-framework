from functools import partial
from os import cpu_count
from typing import List, Optional

import gymnasium.vector
import numpy as np
import pettingzoo
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from supersuit.vector import MakeCPUAsyncConstructor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper


class PettingZooVectorEnv(gymnasium.vector.VectorEnv):
    """
    A PettingZoo ParallelEnv as vectorized environment with one sub-environment ("slot") per possible agent. The
    environment is reset once all agents are done, returning the done flags and rewards of the final step together with
    the observations of the new episode (the last observations are passed as `terminal_observation` infos).

    Agents which leave before the other agents (done, and no longer in `agents`): their done step is passed through,
    so their episode ends with its real last transition. Until the environment is reset, their slot then gets dummy
    steps: an empty observation (zeros), reward 0, truncated (with the empty observation as `terminal_observation`), and
    the info `discard` (not counted as episode, e.g., by the episode logging). The training algorithm still learns from
    these steps: being truncated, their values bootstrap from the empty observation itself, which keeps them
    uninformative, but they take up batch samples.

    Agents which restart on their own while the other agents continue: see `AutoResetSB3VecEnvWrapper`.
    Replaces supersuit's `MarkovVectorEnv`, whose `black_death` option does not support leaving agents.
    """

    def __init__(self, par_env: pettingzoo.ParallelEnv):
        self.par_env = par_env
        self.metadata = par_env.metadata
        self.render_mode = getattr(par_env.unwrapped, "render_mode", None)
        self.possible_agents = list(par_env.possible_agents)
        self.observation_space = par_env.observation_space(self.possible_agents[0])
        self.action_space = par_env.action_space(self.possible_agents[0])
        assert all(
            par_env.observation_space(agent) == self.observation_space for agent in self.possible_agents
        ), "The observation spaces of all agents must be the same."
        assert all(
            par_env.action_space(agent) == self.action_space for agent in self.possible_agents
        ), "The action spaces of all agents must be the same."
        self.num_envs = len(self.possible_agents)
        self._empty_observation = next(
            iter(iterate(self.observation_space, create_empty_array(self.observation_space)))
        )

    def _concatenate_observations(self, observations: dict):
        return concatenate(
            self.observation_space,
            [observations.get(agent, self._empty_observation) for agent in self.possible_agents],
            create_empty_array(self.observation_space, self.num_envs),
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observations, infos = self.par_env.reset(seed=seed, options=options)
        return self._concatenate_observations(observations), [infos.get(agent, {}) for agent in self.possible_agents]

    def step_async(self, actions):
        self._saved_actions = actions

    def step_wait(self):
        return self.step(self._saved_actions)

    def step(self, actions):
        actions = list(iterate(self.action_space, actions))
        active_agents = set(self.par_env.agents)
        observations, rewards, terminations, truncations, infos = self.par_env.step(
            {agent: actions[index] for index, agent in enumerate(self.possible_agents) if agent in active_agents}
        )
        # Agents which were stepped (left agents are not)
        stepped_agents = set(terminations) | set(truncations)
        env_done = all(terminations.get(agent, False) or truncations.get(agent, False) for agent in stepped_agents)

        slot_rewards = np.array([rewards.get(agent, 0.0) for agent in self.possible_agents], dtype=np.float32)
        slot_terminations = np.array([terminations.get(agent, False) for agent in self.possible_agents], dtype=bool)
        slot_truncations = np.array(
            [truncations.get(agent, False) or agent not in stepped_agents for agent in self.possible_agents],
            dtype=bool,
        )
        slot_infos = []
        for agent in self.possible_agents:
            if agent in stepped_agents:
                info = dict(infos.get(agent, {}))
                if env_done:
                    info["terminal_observation"] = observations.get(agent, self._empty_observation)
            else:
                info = {"terminal_observation": self._empty_observation, "discard": True}
            slot_infos.append(info)

        if env_done:
            slot_observations, reset_infos = self.reset()
            slot_infos = [{**info, **reset_info} for info, reset_info in zip(slot_infos, reset_infos)]
        else:
            slot_observations = self._concatenate_observations(observations)
        return slot_observations, slot_rewards, slot_terminations, slot_truncations, slot_infos

    def render(self):
        return self.par_env.render()

    def close(self):
        return self.par_env.close()

    def env_is_wrapped(self, wrapper_class):
        return [False] * self.num_envs


class AutoResetSB3VecEnvWrapper(SB3VecEnvWrapper):
    """
    A SB3VecEnvWrapper for Pettingzoo based vectorized environments (through `PettingZooVectorEnv`).
    `PettingZooVectorEnv` resets the environment automatically once all agents are done and returns the done flags and
    rewards of the final step together with the observations of the new episode.
    Sets infos for each done agent on step (as SB3 expects from auto-resetting vectorized environments):
        - `infos["terminal_observation"]`: the last observation of the finished episode
        - `infos["TimeLimit.truncated"] = True` when truncated (else False)

    For agents which restart on their own, the `final_observation` info (gymnasium's auto-reset convention, see
    `StableBaselinesAgent.train`) is the last observation of the finished episode. It takes precedence, also over the
    `terminal_observation` which `PettingZooVectorEnv` sets to the returned observation when all agents are done.

    Seeding (SB3 calls `seed` when the algorithm gets a `seed`): the seed is applied at the next reset, each PettingZoo
    environment being reset with `seed + index` (by supersuit's vectorized environments).
    """

    def __init__(self, venv):
        super().__init__(venv)
        self._seed_for_next_reset: Optional[int] = None

    def seed(self, seed: Optional[int] = None):
        self._seed_for_next_reset = seed
        return [None if seed is None else seed + index for index in range(self.num_envs)]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        seed = self._seed_for_next_reset if seed is None else seed
        self._seed_for_next_reset = None
        # Note: SB3's vector envs return only observations on reset, and store infos in `self.reset_infos`
        observations, self.reset_infos = self.venv.reset(seed=seed, options=options)
        return observations

    def step_wait(self):
        observations, rewards, terminations, truncations, infos = self.venv.step_wait()
        dones = np.array([terminations[i] or truncations[i] for i in range(len(terminations))])
        for i in np.flatnonzero(dones):
            infos[i]["TimeLimit.truncated"] = bool(truncations[i] and not terminations[i])
            if "final_observation" in infos[i]:
                infos[i]["terminal_observation"] = infos[i]["final_observation"]
            else:
                infos[i].setdefault("terminal_observation", observations[i])
        return observations, rewards, dones, infos


class DiscardingVecMonitor(VecMonitor):
    """VecMonitor which does not report discarded episodes (`discard` info, e.g., the dummy steps of left agents) as
    episodes (no `episode` info, so SB3's `ep_rew_mean` / `ep_len_mean` ignore them)."""

    def step_wait(self):
        observations, rewards, dones, infos = super().step_wait()
        for info in infos:
            if info.get("discard", False):
                info.pop("episode", None)
        return observations, rewards, dones, infos


def pettingzoo_environments_to_vec_env(
    environments: List[pettingzoo.ParallelEnv], max_num_cpus: Optional[int] = None
) -> VecEnv:
    """
    SB3 vectorized environment of PettingZoo environments, with one sub-environment per possible agent of each
    environment (see `PettingZooVectorEnv` and `AutoResetSB3VecEnvWrapper`), monitored.

    Args:
        environments: PettingZoo environments (with the same observation and action spaces).
        max_num_cpus: Maximum number of processes stepping the environments (default: the number of CPUs; at most one
            per environment). With 1, the environments are stepped in this process.
    """
    vector_envs = [PettingZooVectorEnv(environment) for environment in environments]

    def make_env(index: int):
        return vector_envs[index]

    vectorized_environment = MakeCPUAsyncConstructor(min(max_num_cpus or cpu_count(), len(vector_envs)))(
        [partial(make_env, index) for index in range(len(vector_envs))],
        vector_envs[0].observation_space,
        vector_envs[0].action_space,
    )
    return DiscardingVecMonitor(AutoResetSB3VecEnvWrapper(vectorized_environment))
