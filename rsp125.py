from enum import IntEnum

import gymnasium as gym
import numpy as np


class Actions(IntEnum):
    R = 0
    S = 1
    P = 2


class RSP125(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, opp=None, n_history=10, goal=100, render_mode=None):
        self.action_space = gym.spaces.Discrete(3) # グー、チョキ、パー
        self.reward_range = 0.0, 5.0 # 報酬の最小値、最大値
        self.observation_space = gym.spaces.MultiDiscrete((4,) * (2 * n_history))

        self.opp = opp or UniformAgent()
        self.n_history = n_history
        self.goal = goal

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

    def _get_obs(self, opp=False):
        hist = self._action_history[self.game_count:self.game_count+self.n_history, :]
        if opp:
            hist = hist[:, ::-1]
        return hist.ravel()

    def _get_info(self):
        return {'reward_history': self._reward_history[:self.game_count]}

    def _get_reward(self, action0, action1):
        result = (4 + action1 - action0) % 3 - 1 # 負け -1, 引き分け 0, 勝ち 1
        reward0 = [1, 2, 5][action0] if result == +1 else 0
        reward1 = [1, 2, 5][action1] if result == -1 else 0
        return reward0, reward1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.opp.reset(self.np_random)
        self.game_count = 0
        self._action_history = np.full((self.n_history + self.goal, 2), 3, dtype=int)
        self._reward_history = np.zeros((self.goal, 2))
        return self._get_obs(), self._get_info()

    def step(self, action):
        opp_action = self.opp.get_action(self._get_obs(opp=True))
        reward, opp_reward = self._get_reward(action, opp_action)
        self._action_history[self.n_history + self.game_count] = action, opp_action
        self._reward_history[self.game_count] = reward, opp_reward
        self.game_count += 1
        terminated = self.game_count == self.goal
        truncated = False
        self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == 'human':
            action0 = Actions(self._action_history[self.n_history + self.game_count - 1, 0])
            action1 = Actions(self._action_history[self.n_history + self.game_count - 1, 1])
            score0 = self._reward_history[:self.game_count, 0].sum()
            score1 = self._reward_history[:self.game_count, 1].sum()
            print(f'{self.game_count}回目')
            print(f'プレーヤーの手：{action0.name}、相手の手：{action1.name}')
            print(f'プレーヤーのスコア： {score0}、相手のスコア：{score1}')


class UniformAgent:
    def reset(self, rng):
        self.rng = rng

    def get_action(self, obs):
        return self.rng.choice((0, 1, 2), p=(1 / 3, 1 / 3, 1 / 3))


class InputAgent:
    def action(self, obs):
        print('履歴')
        for o in obs.reshape(-1, 2):
            if o[0] < 3:
                print(Actions(o[0]).name, Actions(o[1]).name)
        action = input('R or S or P ? ')
        while action not in Actions.__members__:
            action = input('R or S or P ? ')
        return Actions[action]


def test_run():
    env = RSP125(goal=10, render_mode='human')
    agent = InputAgent()
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated
    env.close()


if __name__ == '__main__':
    test_run()
