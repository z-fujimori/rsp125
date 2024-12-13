from opp_buffer import DQN
# from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from rsp125 import RSP125
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_display import plot_rews,display_percentage_of_hand

def append_act_rew_env0(act_0, rew_0, act_1, rew_1, act_hist, rew_hist):
  for a in act_hist:
    act_0.append([a[0]])   # display_percentage_of_hand()では[x]の形で扱っているため、わざわざ[]で囲って格納してます、、、
    act_1.append([a[1]])
  sum_rew_0 = 0
  sum_rew_1 = 0
  for r in rew_hist:
    sum_rew_0 += r[0]
    sum_rew_1 += r[1]
  rew_0.append(sum_rew_0)
  rew_1.append(sum_rew_1)
  return act_0, rew_0, act_1, rew_1

def append_act_rew_env1(act_0, rew_0, act_1, rew_1, act_hist, rew_hist):
  for a in act_hist:
    act_0.append([a[1]])   # display_percentage_of_hand()では[x]の形で扱っているため、わざわざ[]で囲って格納してます、、、
    act_1.append([a[0]])
  sum_rew_0 = 0
  sum_rew_1 = 0
  for r in rew_hist:
    sum_rew_0 += r[1]
    sum_rew_1 += r[0]
  rew_0.append(sum_rew_0)
  rew_1.append(sum_rew_1)
  return act_0, rew_0, act_1, rew_1

def main(goal=100):
  learn_rate = 0.0005   #  学習りつ DQNのデフォルトは1e-3
  gamma = 0.99    #    割引率   デフォルトは0.99
  num_trials = 700

  act_len_0 = []
  act_len_1 = []
  rew_len_0 = []
  rew_len_1 = []
  env0 = RSP125(goal=100, n_history=5)
  env1 = RSP125(goal=100, n_history=5)
  model0 = DQN(
    "MlpPolicy",
    env0,
    learning_starts=0,
    gradient_steps=-1,
    verbose=0,
    learning_rate=learn_rate,   #  学習りつ
  )
  model1 = DQN(
    "MlpPolicy", 
    env1, 
    learning_starts=0,
    gradient_steps=-1,
    verbose=0,
    learning_rate=learn_rate,   #  学習りつ
  )
  # model0.set_env(RSP125(opp=model1, goal=100))
  # model1.set_env(RSP125(opp=model0, goal=100))
  env0.opp = model1
  env1.opp = model0

  for i in range(num_trials):
    # 学習phase (model0学習 model1固定)
    # # model0.replay_buffer.reset()
    # # model0.gradient_steps = 100
    model0.learn(total_timesteps=1_000, log_interval=100)

    # 評価phase (model0固定 model1固定)
    # # model1.gradient_steps= 0
    # # model1.learn(total_timesteps=1_000, log_interval=100)
    # obs, info  = env1.reset()
    # for k in range(goal):
    #   action = model1.predict(obs, deterministic=True)[0]
    #   obs, reward, terminated, truncated, info = env1.step(action)
    # act_len_0, rew_len_0, act_len_1, rew_len_1 = append_act_rew_env1(act_len_0, rew_len_0, act_len_1, rew_len_1, env1._action_history[5:], env1._reward_history)

    # 学習phase (model0固定 model1学習)
    # # model1.replay_buffer.reset()
    # # model1.gradient_steps= 100
    model1.learn(total_timesteps=1_000, log_interval=100)

    # 評価phase (model0固定 model1固定)
    # # model0.gradient_steps = 0
    # # model0.learn(total_timesteps=1_000, log_interval=100)
    obs, info  = env0.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = env0.step(action)
    act_len_0, rew_len_0, act_len_1, rew_len_1 = append_act_rew_env0(act_len_0, rew_len_0, act_len_1, rew_len_1, env0._action_history[5:], env0._reward_history)

    print(f"i: {i}, reward0: {rew_len_0[i]}, reward1: {rew_len_1[i]}")
  
  output_file_name = 'env0_new_oppDQN_two_player_' + str(learn_rate) + "_" + str(num_trials) + "_1.csv"
  display_percentage_of_hand(act_len_0, act_len_1, output_file_name)
  plot_rews(rew_len_0, rew_len_1, output_file_name, learn_rate)

if __name__ == "__main__":
  main()
