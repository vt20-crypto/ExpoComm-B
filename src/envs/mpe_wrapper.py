"""
MPE (Multi-Agent Particle Environment) wrapper for EPyMARL.

Wraps PettingZoo's MPE environments (e.g., simple_spread_v3) to conform to
the EPyMARL MultiAgentEnv interface, matching the pattern of battle_wrappers.py.

The wrapper handles:
    - PettingZoo parallel API → EPyMARL MultiAgentEnv conversion
    - Observation flattening and padding
    - Reward aggregation (per-agent rewards for value decomposition)
    - Global state construction (concatenation of all observations)
    - Action space normalization

Supported scenarios:
    - simple_spread_v3: Cooperative navigation (N agents, N landmarks)
    - simple_tag_v3: Predator-prey (can be added later)

Usage in YAML config:
    env: "MPE_Spread"
    env_args:
      n_agents: 3          # number of cooperative agents
      max_cycles: 25       # episode length
      continuous_actions: False  # discrete actions for Q-learning
"""

import math
import numpy as np
import torch
from .multiagentenv import MultiAgentEnv


class MPEWrapper(MultiAgentEnv):
    """
    EPyMARL-compatible wrapper for PettingZoo MPE simple_spread_v3.

    In simple_spread, N agents must cooperatively navigate to N landmarks.
    Agents are rewarded based on how close any agent is to each landmark,
    and penalized for collisions. This is a fully cooperative task where
    communication is critical for coordination.

    The wrapper provides:
        - Flat observation vectors for each agent
        - Discrete action space (5 actions: no-op, left, right, up, down)
        - Per-agent rewards (compatible with QMIX value decomposition)
        - Global state = concatenation of all agent observations
    """

    def __init__(self, **env_config):
        self.seed_val = env_config.pop("seed", None)
        self.n_agents_requested = env_config.pop("n_agents", 3)
        self.max_cycles = env_config.pop("max_cycles", 25)
        self.continuous_actions = env_config.pop("continuous_actions", False)

        # Import PettingZoo MPE here to keep it optional
        try:
            from pettingzoo.mpe import simple_spread_v2 as simple_spread_v3
        except ImportError:
            raise ImportError(
                "PettingZoo MPE not found. Install with:\n"
                "  pip install pettingzoo[mpe]\n"
                "Or for older versions:\n"
                "  pip install pettingzoo pygame"
            )

        # Create the parallel environment
        try:
            self._env = simple_spread_v3.parallel_env(
                N=self.n_agents_requested,
                max_cycles=self.max_cycles,
                continuous_actions=self.continuous_actions,
                render_mode=None,
            )
        except TypeError:
            # Older PettingZoo (<=1.14) doesn't support render_mode
            self._env = simple_spread_v3.parallel_env(
                N=self.n_agents_requested,
                max_cycles=self.max_cycles,
                continuous_actions=self.continuous_actions,
            )

        self.episode_limit = self.max_cycles
        self.n_agents = self.n_agents_requested

        # Get agent names (e.g., ['agent_0', 'agent_1', 'agent_2'])
        self.agents = self._env.possible_agents

        # Initialize to get space info
        try:
            self._env.reset(seed=self.seed_val)
        except TypeError:
            self._env.reset()

        # Get observation and action space info from first agent
        # (all agents have identical spaces in simple_spread)
        first_agent = self.agents[0]
        self._obs_space = self._env.observation_space(first_agent)
        self._act_space = self._env.action_space(first_agent)

        # Compute dimensions
        self._obs_size = self._obs_space.shape[0]  # 18 for N=3 (4+4+2N+2N per agent)
        self._n_actions = self._act_space.n  # 5 discrete actions
        self._state_size = self._obs_size * self.n_agents  # concat all obs

        # Internal state
        self._obs = None
        self._episode_steps = 0

    def reset(self):
        """Reset the environment and return initial observations and state."""
        try:
            result = self._env.reset(seed=self.seed_val)
        except TypeError:
            result = self._env.reset()
        # PettingZoo >= 1.22 returns (obs, infos) tuple
        if isinstance(result, tuple):
            obs_dict, _ = result
        else:
            obs_dict = result

        self._obs = [obs_dict[agent].astype(np.float32) for agent in self.agents]
        self._episode_steps = 0

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """
        Execute actions for all agents.

        Args:
            actions: List or tensor of discrete actions, one per agent.

        Returns:
            rewards: List of per-agent rewards.
            terminated: Whether the episode is done.
            info: Additional info dict (empty for MPE).
        """
        # Convert actions to dict format expected by PettingZoo
        action_dict = {}
        for i, agent in enumerate(self.agents):
            if isinstance(actions[i], torch.Tensor):
                action_dict[agent] = actions[i].item()
            elif isinstance(actions[i], np.ndarray):
                action_dict[agent] = int(actions[i])
            else:
                action_dict[agent] = int(actions[i])

        # Step the environment
        # PettingZoo >= 1.22 returns 5 values; older versions return 4
        step_result = self._env.step(action_dict)
        if len(step_result) == 5:
            obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict = step_result
        else:
            obs_dict, rew_dict, terminated_dict, info_dict = step_result
            truncated_dict = {}

        self._episode_steps += 1

        # Process observations (handle terminated agents)
        obs_list = []
        rewards = []
        for agent in self.agents:
            if agent in obs_dict:
                obs_list.append(obs_dict[agent].astype(np.float32))
            else:
                obs_list.append(np.zeros(self._obs_size, dtype=np.float32))

            if agent in rew_dict:
                rewards.append(rew_dict[agent])
            else:
                rewards.append(0.0)

        self._obs = obs_list

        # Check if episode is done
        done = all(
            terminated_dict.get(agent, True) or truncated_dict.get(agent, True)
            for agent in self.agents
        )

        info = {}
        if done or self._episode_steps >= self.episode_limit:
            info["episode_length"] = self._episode_steps
            done = True

        return rewards, done, info

    def get_obs(self):
        """Returns all agent observations in a list of flat numpy arrays."""
        return [obs.copy() for obs in self._obs]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self._obs[agent_id].copy()

    def get_obs_size(self):
        """Returns the shape of a single agent's observation."""
        return self._obs_size

    def get_state(self):
        """
        Returns the global state (concatenation of all observations).

        This is used by QMIX's hypernetwork and the auxiliary state
        prediction task in ExpoComm.
        """
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the global state."""
        return self._state_size

    def get_avail_actions(self):
        """Returns available actions for all agents."""
        return [[1] * self._n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns available actions for agent_id (all actions always available in MPE)."""
        return [1] * self._n_actions

    def get_total_actions(self):
        """Returns the total number of discrete actions."""
        return self._n_actions

    def get_env_info(self):
        """Returns environment info dict expected by EPyMARL."""
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

    def get_stats(self):
        """Returns environment statistics (none for MPE)."""
        return {}

    def render(self, mode="human"):
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Close the environment."""
        self._env.close()

    def seed(self, seed=None):
        """Set the random seed."""
        self.seed_val = seed

    def save_replay(self):
        """Save replay (not supported for MPE)."""
        pass
