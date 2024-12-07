from copy import deepcopy

import numpy as np
from stable_baselines3 import DQN as _DQN

class DQN(_DQN):
  def __init__(self, *args, **kwargs):
    self.opp_replay_buffer = None
    super().__init__(*args, **kwargs)

  def _setup_model(self):
    super()._setup_model()
    if self.opp_replay_buffer is None:
      replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
      self.opp_replay_buffer = self.replay_buffer_class(
        self.buffer_size,
        self.observation_space,
        self.action_space,
        device=self.device,
        n_envs=self.n_envs,
        optimize_memory_usage=self.optimize_memory_usage,
        **replay_buffer_kwargs,
      )

  def _store_transition(
    self, replay_buffer, buffer_action, new_obs, reward, dones, infos
  ):
    opp_buffer_action = new_obs[..., -1]
    for i, info in enumerate(infos):
      if (terminal_obs := info.get("terminal_observation")) is not None:
        opp_buffer_action[i] = terminal_obs[-1]
    batch_size = new_obs.shape[0]
    opp_new_obs = new_obs.reshape(batch_size, -1, 2)[:, :, ::-1].reshape(
      batch_size, -1
    )
    opp_reward = np.array([info["reward_history"][-1, 1] for info in infos])
    opp_dones = deepcopy(dones)
    opp_infos = deepcopy(infos)
    for opp_info, info in zip(opp_infos, infos):
      opp_info["reward_history"] = info["reward_history"][..., ::-1]
    super()._store_transition(
      self.opp_replay_buffer,
      opp_buffer_action,
      opp_new_obs,
      opp_reward,
      opp_dones,
      opp_infos,
    )
    super()._store_transition(
      replay_buffer, buffer_action, new_obs, reward, dones, infos
    )
