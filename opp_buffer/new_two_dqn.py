import numpy as np

from stable_baselines3 import DQN
from rsp125 import RSP125


def main():
  env0 = RSP125(goal=20, n_history=5)
  env1 = RSP125(goal=20, n_history=5)
  model0 = DQN(
    "MlpPolicy",
    env0,
    learning_starts=0,
    gradient_steps=-1,
    batch_size=10,
    #learning_rate=1e-10,
    verbose=0,
  )
  model1 = DQN(
    "MlpPolicy",
    env1,
    learning_starts=0,
    gradient_steps=-1,
    batch_size=10,
    #learning_rate=1e-10,
    verbose=0,
  )
  env0.opp = model1
  env1.opp = model0

  for i in range(2):
    print("\n-----\n")
    model0.learn(total_timesteps=100, log_interval=1)
    model1.learn(total_timesteps=100, log_interval=1)
    model0.replay_buffer.reset()
    model1.replay_buffer.reset()

    obs, info = env0.reset()
    for k in range(20):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = env0.step(action)
      print("action: ",action)
      print("obs: ", obs)
      print("act_his: ",env0._action_history)
      print("rew_his",env0._reward_history)
    # print(np.concatenate([env0._action_history[5:], env0._reward_history], axis=1))
  print(env0._reward_history[19][0])
  print(env0._reward_history[19][1])

if __name__ == "__main__":
  main()