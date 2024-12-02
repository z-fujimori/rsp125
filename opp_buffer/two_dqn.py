from opp_buffer import DQN
from rsp125 import RSP125


def main(goal=100):
  dummy_env = RSP125()
  model0 = DQN(
    "MlpPolicy",
    dummy_env,
    learning_starts=0,
    gradient_steps=0,
    verbose=0,
  )
  model1 = DQN("MlpPolicy", dummy_env)
  model0.set_env(RSP125(opp=model1, goal=100))
  model1.set_env(RSP125(opp=model0, goal=100)) # 実際は使わないので不要かも

  # 初期設定を行うため
  model0.learn(total_timesteps=0)
  model1.learn(total_timesteps=0)
  model1.replay_buffer = model0.opp_replay_buffer

  for i in range(100):
    model0.learn(total_timesteps=1_000, log_interval=1)
    reward0 = model0.replay_buffer.rewards.sum()
    reward1 = model1.replay_buffer.rewards.sum()
    print(f"i: {i}, reward0: {reward0}, reward1: {reward1}")
    model0.train(gradient_steps=1000, batch_size=32)
    model1.train(gradient_steps=1000, batch_size=32)
    model0.replay_buffer.reset()
    model1.replay_buffer.reset()


if __name__ == "__main__":
  main()
    