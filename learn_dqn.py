from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from rsp125 import RSP125
from rsp125 import NashAgent
import numpy as np
import time
import sys
import os
import datetime
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

def main(goal=100):
  start_time = time.time()

  num_trials = 1
  learn_rate = 0.00007   #  学習率 DQNのデフォルトは1e-3
  learn_rate_leverage = 1.0   #  !!!!!!!!!!!!!!!!!!model0がのんびりさんだ、、、、なぜだ
  gamma = 0.99    #    割引率   デフォルトは0.99
  gradient_steps = 1000 # learn()ごとに何回学習するか デフォルトは１ 
  gradient_steps_skale = 1.19
  batch_size = 256 #  default=256
  model0_batch_size = 256
  # gradient_steps × batch_size が1回のトレーニングで使用されるサンプル数
  freq_step = 10
  freq_word = "episode"
  train_freq = (freq_step, freq_word) # 何ステップごとにモデルのトレーニングを行うか default=(1, "step")
  layer = [64,64]
  policy_kwargs = dict(net_arch=layer) # ネットワークのアーキテクチャを変更 デフォルトは[64, 64]
  seed_value = 42 # シードを揃える
  save_model_zip = True

  print(f"num_trials{num_trials} learn_rate{learn_rate} learn_rate_leverage{learn_rate_leverage} gamma{gamma} gradient_steps{gradient_steps} gradient_steps_skale{gradient_steps_skale} batch_size{batch_size} model0_batch_size{model0_batch_size} freq_step{freq_step}{freq_word} seed_value{seed_value} nn_layer{layer}")

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
  act_model1_mod1vsnash = []
  act_nash_mod1vsnash = []
  rew_model1_mod1vsnash = []
  rew_nash_mod1vsnash = []

  env0 = RSP125(goal=100, n_history=5)
  env1 = RSP125(goal=100, n_history=5)
  model0 = DQN(
    "MlpPolicy",
    env0,
    seed=seed_value,
    learning_starts=0,  # 元は0 
    train_freq=train_freq,
    gradient_steps=int(gradient_steps*gradient_steps_skale), 
    verbose=0,
    gamma=gamma,
    batch_size=model0_batch_size,
    learning_rate=learn_rate*learn_rate_leverage,   #  学習率
    # device="cuda", # GPUを使用 "cpu"と書くとCPU使用
    policy_kwargs=policy_kwargs # nnの設定
  )
  model1 = DQN(
    "MlpPolicy", 
    env1, 
    seed=seed_value+1,
    learning_starts=0,  # 元は0
    train_freq=train_freq,
    gradient_steps=gradient_steps,
    verbose=0,
    gamma=gamma,
    batch_size=batch_size,
    learning_rate=learn_rate,   #  学習りつ
    # device="cuda",
    policy_kwargs=policy_kwargs # nnの設定
  )
  # model0.set_env(RSP125(opp=model1, goal=100))
  # model1.set_env(RSP125(opp=model0, goal=100))
  env0.opp = model1
  env1.opp = model0
  # 追加実験用(他の戦略にもロバストか？)
  envNash = RSP125(goal=100, n_history=5, isOppNash=True)

  for i in range(num_trials):
    if i % (num_trials/100) == 0 and i > 1:
      elapsed_time = time.time() - start_time
      remaining_time = elapsed_time*(num_trials-i)/i
      estimated_end_time = time.time() + remaining_time
      print("終了予定: ",datetime.datetime.fromtimestamp(estimated_end_time))
    # 学習phase (model0学習 model1固定)
    # # model0.replay_buffer.reset()
    # # model0.gradient_steps = 100
    model0.learn(total_timesteps=1_000, log_interval=100)

    # 評価phase (model0固定 model1固定)
    # # model1.gradient_steps= 0
    # # model1.learn(total_timesteps=1_000, log_interval=100)
    obs, info  = env1.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = env1.step(action)
    act_len_0_timing1, rew_len_0_timing1, act_len_1_timing1, rew_len_1_timing1 = append_act_rew_env1(act_len_0_timing1, rew_len_0_timing1, act_len_1_timing1, rew_len_1_timing1, env1._action_history[5:], env1._reward_history)
    # 追加実験用(他の戦略にもロバストか？)
    obs, info = envNash.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envNash.step(action)
    act_model1_mod1vsnash, rew_model1_mod1vsnash, act_nash_mod1vsnash, rew_nash_mod1vsnash = append_act_rew_env1(act_model1_mod1vsnash, rew_model1_mod1vsnash, act_nash_mod1vsnash, rew_nash_mod1vsnash, envNash._action_history[5:], envNash._reward_history)

    # 学習phase (model0固定 model1学習)
    # # model1.replay_buffer.reset()
    # # model1.gradient_steps= 100
    model1.learn(total_timesteps=1_000, log_interval=100)

    # 評価phase (model0固定 model1固定)
    # # model0.gradient_steps = 0
    # # model0.learn(total_timesteps=1_000, log_interval=100)
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

    print(f"i: {i} / {num_trials}\ntiming1 reward0: {rew_len_0_timing1[i]}, reward1: {rew_len_1_timing1[i]}\ntiming2 reward0: {rew_len_0_timing2[i]}, reward1: {rew_len_1_timing2[i]}")

  end_time = time.time()
  print(f"Execution time: {end_time - start_time:.2f} seconds")

  local_end_time = time.localtime(end_time)
  format_end_time = time.strftime("%Y-%m%d-%H:%M:%S",local_end_time)

  # 保存用ディレクトリ作成
  result_log_name = f"追加検証_originDQN_mod0*{learn_rate_leverage}-gradient*{gradient_steps_skale}-bach{model0_batch_size}_{format_end_time}_learningRate{learn_rate}_gamma{gamma}_gradientSteps{gradient_steps}_trainFreq{freq_step}{freq_word}_trial{num_trials}_batchSize{batch_size}_nn{str(layer)}_seed{seed_value}"
  os.makedirs(f"./results/{result_log_name}", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/hand_csv", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/rew_plot", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust", exist_ok=True)

  if save_model_zip:
    os.makedirs(f"./model_zips/{result_log_name}", exist_ok=True)
    model0.save(f"./model_zips/{result_log_name}/model0.zip")
    model1.save(f"./model_zips/{result_log_name}/model1.zip")

  all_plot_time = time.time()
  run_time_log = f"all plot {(all_plot_time - start_time)/60:.2f} min\n{result_log_name}"
  display_percentage_of_hand(act_len_0_timing1, act_len_1_timing1, act_len_0_timing2, act_len_1_timing2, result_log_name, run_time_log)
  plot_rews(rew_len_0_timing1, rew_len_1_timing1, rew_len_0_timing2, rew_len_1_timing2, result_log_name, run_time_log, num_trials)

  # # 追加実験用(他の戦略にもロバストか？)
  display_percentage_of_hand(act_model0_mod0vsnash, act_nash_mod0vsnash, act_model1_mod1vsnash, act_nash_mod1vsnash, result_log_name, run_time_log, isRobust=True)
  plot_rews(rew_model0_mod0vsnash, rew_nash_mod0vsnash, rew_model1_mod1vsnash, rew_nash_mod1vsnash, result_log_name, run_time_log, num_trials, isRobust=True)


  all_finish_time = time.time()
  print(f"all finish {(all_finish_time - start_time)/60:.2f} min\n{result_log_name}")

if __name__ == "__main__":
  main()
