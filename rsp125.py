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
        # self.hand_percentafe_space_between = 45
        # self.hand_percentafe_space = [[0, 0, 0] for _ in range(222)]
        # self.hand_percentafe_space_count = [0, 0] # カウント, 初めの10回か
        self.action_history = np.full((n_history + goal, 2), 3, dtype=int)

        self.opp = opp or UniformAgent()
        # self.opp = opp or TitForTat()
        self.n_history = n_history
        self.goal = goal

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

    def _get_obs(self, opp=False):
        hist = self.action_history[self.game_count:self.game_count+self.n_history, :]
        if opp:
            hist = hist[:, ::-1]
        return hist.ravel()

    def _get_info(self):
        return {'reward_history': self._reward_history[:self.game_count]}
    
    def get_hist_info(self):
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
        self.action_history = np.full((self.n_history + self.goal, 2), 3, dtype=int)
        self._reward_history = np.zeros((self.goal, 2))
        # self.hand_percentafe_space = [[0, 0, 0] for _ in range(222)]
        return self._get_obs(), self._get_info()

    def step(self, action):
        opp_action = self.opp.get_action(self._get_obs(opp=True))
        reward, opp_reward = self._get_reward(action, opp_action)
        self.action_history[self.n_history + self.game_count] = action, opp_action
        self._reward_history[self.game_count] = reward, opp_reward
        self.game_count += 1
        terminated = self.game_count == self.goal
        truncated = False

        # if self.game_count >= 10 :
        #     self.hand_percentafe_space[(self.game_count-10) // self.hand_percentafe_space_between][]
        # print("-----")

        self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == 'human':
            action0 = Actions(self.action_history[self.n_history + self.game_count - 1, 0])
            action1 = Actions(self.action_history[self.n_history + self.game_count - 1, 1])
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
    
class NashAgent(UniformAgent):
    def get_action(self, obs):
        return self.rng.choice((0, 1, 2), p=(2 / 17, 10 / 17, 5 / 17))
    
class TitForTat(NashAgent):
    def get_hand(self, obs):
        rng = np.random.default_rng()
        last_strategy_hand = obs[-1]
        last_opponent_hand = obs[-2]
        second_last_opponent_hand = obs[-4]
        therd_last_opponent_hand = obs[-6]
        forth_last_opponent_hand = obs[-8]
    # 繰り返し対策
        if (last_opponent_hand == second_last_opponent_hand) and (last_opponent_hand == therd_last_opponent_hand) and (last_opponent_hand == forth_last_opponent_hand):
            if rng.random() >= 1/8:
                return win_hand(last_opponent_hand)
        elif (last_opponent_hand == second_last_opponent_hand) and (last_opponent_hand == therd_last_opponent_hand) :
            if rng.random() >= 1/4:
                return win_hand(last_opponent_hand)
        elif (last_opponent_hand == second_last_opponent_hand):
            if rng.random() >= 1/2:
                return win_hand(last_opponent_hand)
        if last_strategy_hand == 2 and last_opponent_hand == 0:
            return Actions["R"]
        elif last_strategy_hand == 0 and last_opponent_hand == 2:
            return Actions["P"]
        else:
            return self.np_random.choice(self.hands, p=self.ratio)

def win_hand(hand):
    if hand == 0:
        return Actions["P"]
    elif hand == 1:
        return Actions["R"]
    else:
        return Actions["S"]

def test_run():
    env = RSP125(goal=10, render_mode='human')
    agent = InputAgent()
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()
        done = terminated or truncated
    env.close()


if __name__ == '__main__':
    test_run()
