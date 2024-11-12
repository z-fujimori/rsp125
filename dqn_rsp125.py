from stable_baselines3 import DQN

from rsp125 import RSP125
from rsp125 import UniformAgent

def main():
    env = RSP125(goal=100)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save('dqn_rsp125') # 学習ずみのモデルを別保存
    hist = model.observation_space
    return model, hist

# def test():
#     env = RSP125(goal=100)
#     model = DQN.load('dqn_rsp125')
#     obs, info = env.reset()
#     for i in range(10_000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         if terminated or truncated:
#             obs, info = env.reset()

def rally():
    print("start rally")
    env = RSP125(goal=100)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=400)
    # model.save('dqn_rsp125') # 学習ずみのモデルを別保存
    for i in range(10):
        model.set_env(env)
        model.learn(total_timesteps=400, reset_num_timesteps=False)
        print(env.action_history)
    model.save('dqn_rsp125') # 学習ずみのモデルを別保存
    return model

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

if __name__ == '__main__':
    # main()
    # mod, hist = main()
    mod = rally()
    obs = [1, 0, 1, 0, 2, 1, 1, 1, 0, 1, 0, 2, 1, 1, 2, 0, 1, 1, 1, 2]
    action, _states = mod.predict(obs, deterministic=True)
    # print("hist shape   ", hist)
    print(action)
    print(mod._last_obs)
    print(_states)
    # print(model.get_action([1,2,1,1,1,0,1,2,3,2]))
