from stable_baselines3 import DQN

from rsp125 import RSP125


def main():
    env = RSP125(goal=100)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save('dqn_rsp125')
    del model


def test():
    env = RSP125(goal=100)
    model = DQN.load('dqn_rsp125')
    obs, info = env.reset()
    for i in range(1_000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == '__main__':
    main()
