from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from rsp125 import RSP125
from rsp125 import UniformAgent

def main():
    env = RSP125(goal=100)
    model = DQN('MlpPolicy', env, verbose=1)
    action_callback = FullActionHistoryCallback()
    model.learn(total_timesteps=10_000, callback=action_callback)
    model.save('dqn_rsp125') # 学習ずみのモデルを別保存
    action_history = action_callback.action_history
    hist = model.observation_space
    return model, hist, action_history

# def test():
#     env = RSP125(goal=100)
#     model = DQN.load('dqn_rsp125')
#     obs, info = env.reset()
#     for i in range(10_000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             obs, info = env.reset()
def test_rally():
    env = RSP125(goal=100)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save('dqn_rsp125') # 学習ずみのモデルを別保存

    print("\n************************\n")
    
    class Opp(UniformAgent):
        def get_action(self, obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

    env2 = RSP125(opp=Opp(), goal=100)
    model2 = DQN('MlpPolicy', env2, verbose=1)
    model2.learn(total_timesteps=10_000)
    model2.save('dqn_rsp125')
    print("\n************************\n")

    class Opp(UniformAgent):
        def get_action(self, obs):
            action, _ = model2.predict(obs, deterministic=True)
            return action

    env = RSP125(opp=Opp(), goal=100)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save('dqn_rsp125') # 学習ずみのモデルを別保存
    print("\n**\n")

    return model

# 学習中のアクション履歴を記録するカスタムコールバック
class FullActionHistoryCallback(BaseCallback):
    def __init__(self):
        super(FullActionHistoryCallback, self).__init__()
        self.action_history = []

    def _on_step(self) -> bool:
        if self.locals.get('actions') is not None:
            action = self.locals['actions']
            obs = self.locals['new_obs']
            if hasattr(self.training_env.envs[0], 'opp'):
                opp_action = self.training_env.envs[0].opp.get_action(obs)
                self.action_history.append((action, opp_action))
        return True

def rally():
    print("start rally")
    env = RSP125(goal=100)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=400)
    # model.save('dqn_rsp125') # 学習ずみのモデルを別保存
    for i in range(10):
        model.set_env(env)
        model.learn(total_timesteps=100, reset_num_timesteps=False)
        # print(env.action_history)
    model.save('dqn_rsp125') # 学習ずみのモデルを別保存
    return model


def create_new_agent(base_class, mod_action, class_name):
    return type(
        class_name,  # クラス名
        (base_class,),  # 継承するベースクラス
        {
            # get_actionメソッドを上書き
            "get_action": lambda self, obs: mod_action.predict(obs, deterministic=True)
        },
    )

def two_player():
    env = RSP125(goal=100)
    playerA = DQN('MlpPolicy', env, verbose=1)
    playerB = DQN('MlpPolicy', env, verbose=1)
    playerA.learn(total_timesteps=2000)
    playerB.learn(total_timesteps=2000)


if __name__ == '__main__':
    # main()
    mod, hist, action_history = main()
    print(action_history)
    print("長さ: ",len(action_history))
    print(f"総履歴数: {len(action_history)}")

    # 初めの3つの100回分を表示
    chunk_size = 100
    # 1回目の100回
    print("1回目の100回:")
    for i, (player_action, opp_action) in enumerate(action_history[:chunk_size], 1):
        print(f"{i}回目: プレイヤー={player_action}, 相手={opp_action}")
    # 2回目の100回
    print("\n2回目の100回:")
    start_idx = chunk_size
    for i, (player_action, opp_action) in enumerate(action_history[start_idx:start_idx + chunk_size], 1):
        print(f"2: {i}回目: プレイヤー={player_action}, 相手={opp_action}")
    # 3回目の100回
    print("\n3回目の100回:")
    start_idx = 2 * chunk_size
    for i, (player_action, opp_action) in enumerate(action_history[start_idx:start_idx + chunk_size], 1):
        print(f"3: {i}回目: プレイヤー={player_action}, 相手={opp_action}")
    
    # 終盤の100回
    print("\n99回目の100回:")
    start_idx = 99 * chunk_size
    for i, (player_action, opp_action) in enumerate(action_history[start_idx:start_idx + chunk_size], 1):
        print(f"99: {i}回目: プレイヤー={player_action}, 相手={opp_action}")
    # # mod = rally()
    # obs = [1, 0, 1, 0, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1, 2, 0, 1, 1, 1, 2]
    # action, _states = mod.predict(obs, deterministic=True)
    # # print("hist shape   ", hist)
    # print(action)
    # print(mod._last_obs)
    # print(_states)
    # print(model.get_action([1,2,1,1,1,0,1,2,3,2]))
