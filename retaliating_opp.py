# from opp_buffer import DQN
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from rsp125 import RSP125
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_display import plot_rews,display_percentage_of_hand,retaliating_plot_rews

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

seed = 100
def main(goal=100):
  start_time = time.time()


  num_trials = 1000
  learn_rate = 0.0005   #  学習率 DQNのデフォルトは1e-3
  gamma = 0.99    #    割引率   デフォルトは0.99
  gradient_steps = 100 # learn()ごとに何回学習するか デフォルトは１ 
  batch_size = 100 #  default=256
  # gradient_steps × batch_size が1回のトレーニングで使用されるサンプル数
  freq_step = 10
  freq_word = "episode"
  train_freq = (freq_step, freq_word) # 何ステップごとにモデルのトレーニングを行うか default=(1, "step")
  seed_value = 42 # シードを揃える
  save_model_zip = True

  print(f"しっぺ返しと対戦[β] num_trials{num_trials} learn_rate{learn_rate} gamma{gamma} gradient_steps{gradient_steps} batch_size{batch_size} freq_step{freq_step}{freq_word} seed_value{seed_value}")

  act_len_0_timing1 = []
  act_len_1_timing1 = []
  rew_len_0_timing1 = []
  rew_len_1_timing1 = []
  act_len_0_timing2 = []
  act_len_1_timing2 = []
  rew_len_0_timing2 = []
  rew_len_1_timing2 = []
  # 追加実験用(他の戦略にもロバストか？)
  act_model0_mod0vsnash = []
  act_nash_mod0vsnash = []
  rew_model0_mod0vsnash = []
  rew_nash_mod0vsnash = []
  act_model0_mod0vsuniform = []
  act_uniform_mod0vsuniform = []
  rew_model0_mod0vsuniform = []
  rew_uniform_mod0vsuniform = []
  act_model0_mod0vsR = []    # Rしか出さない
  act_R_mod0vsR = []
  rew_model0_mod0vsR = []
  rew_R_mod0vsR = []
  act_model1_mod1vsR = []
  act_R_mod1vsR = []
  rew_model1_mod1vsR = []
  rew_R_mod1vsR = []
  act_model0_mod0vsC = []    # Cしか出さない
  act_C_mod0vsC = []
  rew_model0_mod0vsC = []
  rew_C_mod0vsC = []
  act_model1_mod1vsC = []
  act_C_mod1vsC = []
  rew_model1_mod1vsC = []
  rew_C_mod1vsC = []
  act_model0_mod0vsP = []    # Pしか出さない
  act_P_mod0vsP = []
  rew_model0_mod0vsP = []
  rew_P_mod0vsP = []
  act_model1_mod1vsP = []
  act_P_mod1vsP = []
  rew_model1_mod1vsP = []
  rew_P_mod1vsP = []

  env0 = RSP125(goal=100, n_history=5)  #  oppType="Nash" を追加すると 相手がtit for tat からnashに
  model0 = DQN(
    "MlpPolicy",
    env0,
    seed=seed_value,
    learning_starts=0,  # 元は0 
    train_freq=train_freq,
    gradient_steps=gradient_steps, 
    verbose=0,
    learning_rate=learn_rate, #  学習率
    # device="cuda", # GPUを使用 "cpu"と書くとCPU使用
  )
  # 追加実験用(他の戦略にもロバストか？)
  envNash = RSP125(goal=100, n_history=5, oppType="Nash")
  envUniform = RSP125(goal=100, n_history=5, oppType="Uniform")
  envR = RSP125(goal=100, n_history=5, oppType="R")
  envC = RSP125(goal=100, n_history=5, oppType="C")
  envP = RSP125(goal=100, n_history=5, oppType="P")

  for i in range(num_trials):
    # 学習phase (model0学習 model1固定)
    # # model0.replay_buffer.reset()
    # # model0.gradient_steps = 100
    model0.learn(total_timesteps=1_000, log_interval=100)

    obs, info  = env0.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]       # ！！！！！！！！！ここは互い違いでなければいけない？
      obs, reward, terminated, truncated, info = env0.step(action)
    act_len_0_timing2, rew_len_0_timing2, act_len_1_timing2, rew_len_1_timing2 = append_act_rew_env0(act_len_0_timing2, rew_len_0_timing2, act_len_1_timing2, rew_len_1_timing2, env0._action_history[5:], env0._reward_history)
    
    # 追加実験用(他の戦略にもロバストか？)
    obs, info = envNash.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envNash.step(action)
    act_model0_mod0vsnash, rew_model0_mod0vsnash, act_nash_mod0vsnash, rew_nash_mod0vsnash = append_act_rew_env0(act_model0_mod0vsnash, rew_model0_mod0vsnash, act_nash_mod0vsnash, rew_nash_mod0vsnash, envNash._action_history[5:], envNash._reward_history)
    obs, info = envUniform.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envUniform.step(action)
    act_model0_mod0vsuniform, rew_model0_mod0vsuniform, act_uniform_mod0vsuniform, rew_uniform_mod0vsuniform = append_act_rew_env0(act_model0_mod0vsuniform, rew_model0_mod0vsuniform, act_uniform_mod0vsuniform, rew_uniform_mod0vsuniform, envUniform._action_history[5:], envUniform._reward_history)
    obs, info = envR.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envR.step(action)
    act_model0_mod0vsR, rew_model0_mod0vsR, act_R_mod0vsR, rew_R_mod0vsR = append_act_rew_env0(act_model0_mod0vsR, rew_model0_mod0vsR, act_R_mod0vsR, rew_R_mod0vsR, envR._action_history[5:], envR._reward_history)
    obs, info = envC.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envC.step(action)
    act_model0_mod0vsC, rew_model0_mod0vsC, act_C_mod0vsC, rew_C_mod0vsC = append_act_rew_env0(act_model0_mod0vsC, rew_model0_mod0vsC, act_C_mod0vsC, rew_C_mod0vsC, envC._action_history[5:], envC._reward_history)
    obs, info = envP.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envP.step(action)
    act_model0_mod0vsP, rew_model0_mod0vsP, act_P_mod0vsP, rew_P_mod0vsP = append_act_rew_env0(act_model0_mod0vsP, rew_model0_mod0vsP, act_P_mod0vsP, rew_P_mod0vsP, envP._action_history[5:], envP._reward_history)

    print(f"i: {i} / {num_trials}\ntiming2 reward0: {rew_len_0_timing2[i]}, reward1: {rew_len_1_timing2[i]}")

  end_time = time.time()
  print(f"Execution time: {end_time - start_time:.2f} seconds")

  local_end_time = time.localtime(end_time)
  format_end_time = time.strftime("%Y-%m%d-%H:%M:%S",local_end_time)

  # 保存用ディレクトリ作成
  result_log_name = f"対しっぺ検証(rsp,nash,uni)_{format_end_time}_learningRate{learn_rate}_gamma{gamma}_gradientSteps{gradient_steps}_trainFreq{freq_step}{freq_word}_trial{num_trials}_batchSize{batch_size}_seed{seed_value}"
  os.makedirs(f"./results/{result_log_name}", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/hand_csv", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/rew_plot", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/nash", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/uniform", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/R", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/C", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/P", exist_ok=True)

  if save_model_zip:
    os.makedirs(f"./model_zips/{result_log_name}", exist_ok=True)
    model0.save(f"./model_zips/{result_log_name}/model0.zip")

  display_percentage_of_hand(act_len_0_timing1, act_len_1_timing1, act_len_0_timing2, act_len_1_timing2, result_log_name)
  retaliating_plot_rews(rew_len_0_timing1, rew_len_1_timing1, rew_len_0_timing2, rew_len_1_timing2, result_log_name, learn_rate, num_trials)

    # # 追加実験用(他の戦略にもロバストか？)
  display_percentage_of_hand([], [], act_model0_mod0vsnash, act_nash_mod0vsnash, result_log_name, oppType="Nash")
  retaliating_plot_rews([], [], rew_model0_mod0vsnash, rew_nash_mod0vsnash, result_log_name, learn_rate, num_trials, oppType="Nash")
  display_percentage_of_hand([], [], act_model0_mod0vsuniform, act_uniform_mod0vsuniform, result_log_name, oppType="Uniform")
  retaliating_plot_rews([], [], rew_model0_mod0vsuniform, rew_uniform_mod0vsuniform, result_log_name, learn_rate, num_trials, oppType="Uniform")
  display_percentage_of_hand(act_model0_mod0vsR, act_R_mod0vsR, act_model1_mod1vsR, act_R_mod1vsR, result_log_name, oppType="R")
  plot_rews(rew_model0_mod0vsR, rew_R_mod0vsR, rew_model1_mod1vsR, rew_R_mod1vsR, result_log_name, num_trials, oppType="R")
  display_percentage_of_hand(act_model0_mod0vsC, act_C_mod0vsC, act_model1_mod1vsC, act_C_mod1vsC, result_log_name, oppType="C")
  plot_rews(rew_model0_mod0vsC, rew_C_mod0vsC, rew_model1_mod1vsC, rew_C_mod1vsC, result_log_name, num_trials, oppType="C")
  display_percentage_of_hand(act_model0_mod0vsP, act_P_mod0vsP, act_model1_mod1vsP, act_P_mod1vsP, result_log_name, oppType="P")
  plot_rews(rew_model0_mod0vsP, rew_P_mod0vsP, rew_model1_mod1vsP, rew_P_mod1vsP, result_log_name, num_trials, oppType="P")

  all_finish_time = time.time()
  print(f"all finish {(all_finish_time - start_time)/60:.2f} min\n{result_log_name}")

if __name__ == "__main__":
  main()
