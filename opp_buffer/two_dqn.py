from opp_buffer import DQN
from stable_baselines3.common.logger import configure
from rsp125 import RSP125
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_display import plot_rews,display_percentage_of_hand

logger = configure("logs/custom_logs", ["stdout", "csv"])

learn_rate = 0.0005   #  学習りつ DQNのデフォルトは1e-3
gamma = 0.99    #    割引率   デフォルトは0.99
learn_count = 1

def main(goal=100):
  act_len_0 = []
  act_len_1 = []
  rew_len_0 = []
  rew_len_1 = []
  dummy_env = RSP125()
  model0 = DQN(
    "MlpPolicy",
    dummy_env,
    learning_starts=0,
    gradient_steps=0,
    verbose=0,
    learning_rate=learn_rate   #  学習りつ
  )
  model1 = DQN("MlpPolicy", dummy_env, learning_rate=learn_rate)   #  学習りつ
  model0.set_env(RSP125(opp=model1, goal=100))
  model1.set_env(RSP125(opp=model0, goal=100)) # 実際は使わないので不要かも
  model0.set_logger(logger)

  # 初期設定を行うため
  model0.learn(total_timesteps=0)
  model1.learn(total_timesteps=0)
  model1.replay_buffer = model0.opp_replay_buffer
  print(model0.replay_buffer.size)

  for i in range(learn_count):
    rew_0 = 0
    rew_1 = 0
    model0.learn(total_timesteps=1_000, log_interval=100) # log_interval を調整で表示間隔
    reward0 = model0.replay_buffer.rewards.sum()
    # reward1 = model1.replay_buffer.rewards.sum()  ここに疑問？
    reward1 = model0.opp_replay_buffer.rewards.sum()

    print(model0.replay_buffer)
    print(model0.opp_replay_buffer)

    print(f"i: {i}, reward0: {reward0}, reward1: {reward1}")
    model0.logger.record("reward/player_0", reward0)
    model0.logger.record("reward/player_1", reward1)
    model0.logger.record("time/step", i)
    model0.logger.dump(step=i) # 出力
    # actionを記録
    for action in model0.replay_buffer.actions[:1000]:
      act_len_0.append(action[0])  # [[2]],,,,の形で入っているが、display_percentage_of_hand()で[2] の形で扱っているため
    for action in model1.replay_buffer.actions[:1000]:
      act_len_1.append(action[0])
    # 報酬の合計を記録
    for rew in model0.replay_buffer.rewards[:1000]:
      rew_0 += rew
    for rew in model1.replay_buffer.rewards[:1000]:
      rew_1 += rew
    rew_len_0.append(rew_0)
    rew_len_1.append(rew_1)

    model0.train(gradient_steps=1000, batch_size=32)
    model1.train(gradient_steps=1000, batch_size=32)
    model0.replay_buffer.reset()
    model1.replay_buffer.reset()
  
  print(model0.replay_buffer)
  print(len(model0.replay_buffer.rewards))
  model0.logger.record("custom_metric", 42)
  # print(model0.replay_data)
  output_file_name = 'two_player_' + str(learn_rate) + "_" + str(learn_count) + "_1.csv"
  display_percentage_of_hand(act_len_0, act_len_1, output_file_name)
  plot_rews(rew_len_0, rew_len_1, "id", learn_rate)
  

if __name__ == "__main__":
  main()
