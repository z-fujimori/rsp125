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
  start_time = time.time()

  num_trials = 10000
  learn_rate = 0.00007   #  学習率 DQNのデフォルトは1e-3
  learn_rate_leverage = 1.0   #  !!!!!!!!!!!!!!!!!!model0がのんびりさんだ、、、、なぜだ
  gamma = 0.99    #    割引率   デフォルトは0.99
  gradient_steps = 1000 # learn()ごとに何回学習するか デフォルトは１ 
  gradient_steps_skale = 1.0
  batch_size = 256 #  default=256
  model0_batch_size = 256
  # gradient_steps × batch_size が1回のトレーニングで使用されるサンプル数
  freq_step = 10
  freq_word = "episode"
  train_freq = (freq_step, freq_word) # 何ステップごとにモデルのトレーニングを行うか default=(1, "step")
  layer = [64,64]
  policy_kwargs = dict(net_arch=layer) # ネットワークのアーキテクチャを変更 デフォルトは[64, 64]
  seed_value = 82 # シードを揃える
  save_model_zip = True
  n_history = 5

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
  act_model0_mod0vsNash = []
  act_Nash_mod0vsNash = []
  rew_model0_mod0vsNash = []
  rew_Nash_mod0vsNash = []
  act_model1_mod1vsNash = []
  act_Nash_mod1vsNash = []
  rew_model1_mod1vsNash = []
  rew_Nash_mod1vsNash = []
  act_model0_mod0vsUniform = []
  act_Uniform_mod0vsUniform = []
  rew_model0_mod0vsUniform = []
  rew_Uniform_mod0vsUniform = []
  act_model1_mod1vsUniform = []
  act_Uniform_mod1vsUniform = []
  rew_model1_mod1vsUniform = []
  rew_Uniform_mod1vsUniform = []
  act_model0_mod0vsTendR = []    # Rを出しがち
  act_TendR_mod0vsTendR = []
  rew_model0_mod0vsTendR = []
  rew_TendR_mod0vsTendR = []
  act_model1_mod1vsTendR = []
  act_TendR_mod1vsTendR = []
  rew_model1_mod1vsTendR = []
  rew_TendR_mod1vsTendR = []
  act_model0_mod0vsTendC = []    # Cを出しがち
  act_TendC_mod0vsTendC = []
  rew_model0_mod0vsTendC = []
  rew_TendC_mod0vsTendC = []
  act_model1_mod1vsTendC = []
  act_TendC_mod1vsTendC = []
  rew_model1_mod1vsTendC = []
  rew_TendC_mod1vsTendC = []
  act_model0_mod0vsTendP = []    # Pを出しがち
  act_TendP_mod0vsTendP = []
  rew_model0_mod0vsTendP = []
  rew_TendP_mod0vsTendP = []
  act_model1_mod1vsTendP = []
  act_TendP_mod1vsTendP = []
  rew_model1_mod1vsTendP = []
  rew_TendP_mod1vsTendP = []
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

  env0 = RSP125(goal=100, n_history=n_history)
  env1 = RSP125(goal=100, n_history=n_history)
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
  envNash = RSP125(goal=100, n_history=n_history, oppType="Nash")
  envUniform = RSP125(goal=100, n_history=n_history, oppType="Uniform")
  envTendR = RSP125(goal=100, n_history=n_history, oppType="TendR")
  envTendC = RSP125(goal=100, n_history=n_history, oppType="TendC")
  envTendP = RSP125(goal=100, n_history=n_history, oppType="TendP")
  envR = RSP125(goal=100, n_history=n_history, oppType="R")
  envC = RSP125(goal=100, n_history=n_history, oppType="C")
  envP = RSP125(goal=100, n_history=n_history, oppType="P")

  for i in range(num_trials):
    if i % (num_trials/100) == 0 and i > 1:
      elapsed_time = time.time() - start_time
      remaining_time = elapsed_time*(num_trials-i)/i
      estimated_end_time = time.time() + remaining_time
      print("終了予定: ",datetime.datetime.fromtimestamp(estimated_end_time))
    # model0学習
    model0.learn(total_timesteps=1_000, log_interval=100)

    # model1に行動させた評価
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
    act_model1_mod1vsNash, rew_model1_mod1vsNash, act_Nash_mod1vsNash, rew_Nash_mod1vsNash = append_act_rew_env0(act_model1_mod1vsNash, rew_model1_mod1vsNash, act_Nash_mod1vsNash, rew_Nash_mod1vsNash, envNash._action_history[5:], envNash._reward_history)
    obs, info = envUniform.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envUniform.step(action)
    act_model1_mod1vsUniform, rew_model1_mod1vsUniform, act_Uniform_mod1vsUniform, rew_Uniform_mod1vsUniform = append_act_rew_env0(act_model1_mod1vsUniform, rew_model1_mod1vsUniform, act_Uniform_mod1vsUniform, rew_Uniform_mod1vsUniform, envUniform._action_history[5:], envUniform._reward_history)
    obs, info = envTendR.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envTendR.step(action)
    act_model1_mod1vsTendR, rew_model1_mod1vsTendR, act_TendR_mod1vsTendR, rew_TendR_mod1vsTendR = append_act_rew_env0(act_model1_mod1vsTendR, rew_model1_mod1vsTendR, act_TendR_mod1vsTendR, rew_TendR_mod1vsTendR, envTendR._action_history[5:], envTendR._reward_history)
    obs, info = envTendC.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envTendC.step(action)
    act_model1_mod1vsTendC, rew_model1_mod1vsTendC, act_TendC_mod1vsTendC, rew_TendC_mod1vsTendC = append_act_rew_env0(act_model1_mod1vsTendC, rew_model1_mod1vsTendC, act_TendC_mod1vsTendC, rew_TendC_mod1vsTendC, envTendC._action_history[5:], envTendC._reward_history)
    obs, info = envTendP.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envTendP.step(action)
    act_model1_mod1vsTendP, rew_model1_mod1vsTendP, act_TendP_mod1vsTendP, rew_TendP_mod1vsTendP = append_act_rew_env0(act_model1_mod1vsTendP, rew_model1_mod1vsTendP, act_TendP_mod1vsTendP, rew_TendP_mod1vsTendP, envTendP._action_history[5:], envTendP._reward_history)
    obs, info = envR.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envR.step(action)
    act_model1_mod1vsR, rew_model1_mod1vsR, act_R_mod1vsR, rew_R_mod1vsR = append_act_rew_env0(act_model1_mod1vsR, rew_model1_mod1vsR, act_R_mod1vsR, rew_R_mod1vsR, envR._action_history[5:], envR._reward_history)
    obs, info = envC.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envC.step(action)
    act_model1_mod1vsC, rew_model1_mod1vsC, act_C_mod1vsC, rew_C_mod1vsC = append_act_rew_env0(act_model1_mod1vsC, rew_model1_mod1vsC, act_C_mod1vsC, rew_C_mod1vsC, envC._action_history[5:], envC._reward_history)
    obs, info = envP.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envP.step(action)
    act_model1_mod1vsP, rew_model1_mod1vsP, act_P_mod1vsP, rew_P_mod1vsP = append_act_rew_env0(act_model1_mod1vsP, rew_model1_mod1vsP, act_P_mod1vsP, rew_P_mod1vsP, envP._action_history[5:], envP._reward_history)

    # model1学習
    model1.learn(total_timesteps=1_000, log_interval=100)

    # model0に行動させた評価
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
    act_model0_mod0vsNash, rew_model0_mod0vsNash, act_Nash_mod0vsNash, rew_Nash_mod0vsNash = append_act_rew_env0(act_model0_mod0vsNash, rew_model0_mod0vsNash, act_Nash_mod0vsNash, rew_Nash_mod0vsNash, envNash._action_history[5:], envNash._reward_history)
    obs, info = envUniform.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envUniform.step(action)
    act_model0_mod0vsUniform, rew_model0_mod0vsUniform, act_Uniform_mod0vsUniform, rew_Uniform_mod0vsUniform = append_act_rew_env0(act_model0_mod0vsUniform, rew_model0_mod0vsUniform, act_Uniform_mod0vsUniform, rew_Uniform_mod0vsUniform, envUniform._action_history[5:], envUniform._reward_history)
    obs, info = envTendR.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envTendR.step(action)
    act_model0_mod0vsTendR, rew_model0_mod0vsTendR, act_TendR_mod0vsTendR, rew_TendR_mod0vsTendR = append_act_rew_env0(act_model0_mod0vsTendR, rew_model0_mod0vsTendR, act_TendR_mod0vsTendR, rew_TendR_mod0vsTendR, envTendR._action_history[5:], envTendR._reward_history)
    obs, info = envTendC.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envTendC.step(action)
    act_model0_mod0vsTendC, rew_model0_mod0vsTendC, act_TendC_mod0vsTendC, rew_TendC_mod0vsTendC = append_act_rew_env0(act_model0_mod0vsTendC, rew_model0_mod0vsTendC, act_TendC_mod0vsTendC, rew_TendC_mod0vsTendC, envTendC._action_history[5:], envTendC._reward_history)
    obs, info = envTendP.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = envTendP.step(action)
    act_model0_mod0vsTendP, rew_model0_mod0vsTendP, act_TendP_mod0vsTendP, rew_TendP_mod0vsTendP = append_act_rew_env0(act_model0_mod0vsTendP, rew_model0_mod0vsTendP, act_TendP_mod0vsTendP, rew_TendP_mod0vsTendP, envTendP._action_history[5:], envTendP._reward_history)
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

    print(f"i: {i} / {num_trials}\ntiming1 reward0: {rew_len_0_timing1[i]}, reward1: {rew_len_1_timing1[i]}\ntiming2 reward0: {rew_len_0_timing2[i]}, reward1: {rew_len_1_timing2[i]}")

  end_time = time.time()
  print(f"Execution time: {end_time - start_time:.2f} seconds")

  local_end_time = time.localtime(end_time)
  format_end_time = time.strftime("%Y-%m%d-%H:%M:%S",local_end_time)

  # 保存用ディレクトリ作成
  result_log_name = f"seed82(rsp,nash,uni,tend)_mod0*{learn_rate_leverage}-gradient*{gradient_steps_skale}-bach{model0_batch_size}_{format_end_time}_learningRate{learn_rate}_gamma{gamma}_gradientSteps{gradient_steps}_trainFreq{freq_step}{freq_word}_trial{num_trials}_batchSize{batch_size}_nn{str(layer)}_seed{seed_value}_history{n_history}"
  os.makedirs(f"./results/{result_log_name}", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/hand_csv", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/rew_plot", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/nash", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/uniform", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/tendR", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/tendC", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/tendP", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/R", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/C", exist_ok=True)
  os.makedirs(f"./results/{result_log_name}/robust/P", exist_ok=True)

  if save_model_zip:
    os.makedirs(f"./model_zips/{result_log_name}", exist_ok=True)
    model0.save(f"./model_zips/{result_log_name}/model0.zip")
    model1.save(f"./model_zips/{result_log_name}/model1.zip")

  all_plot_time = time.time()
  run_time_log = f"all plot {(all_plot_time - start_time)/60:.2f} min\n{result_log_name}"
  display_percentage_of_hand(act_len_0_timing1, act_len_1_timing1, act_len_0_timing2, act_len_1_timing2, result_log_name, run_time_log)
  plot_rews(rew_len_0_timing1, rew_len_1_timing1, rew_len_0_timing2, rew_len_1_timing2, result_log_name, run_time_log, num_trials)

  # # 追加実験用(他の戦略にもロバストか？)
  display_percentage_of_hand(act_model0_mod0vsNash, act_Nash_mod0vsNash, act_model1_mod1vsNash, act_Nash_mod1vsNash, result_log_name, run_time_log, oppType="Nash")
  plot_rews(rew_model0_mod0vsNash, rew_Nash_mod0vsNash, rew_model1_mod1vsNash, rew_Nash_mod1vsNash, result_log_name, run_time_log, num_trials, oppType="Nash")
  display_percentage_of_hand(act_model0_mod0vsUniform, act_Uniform_mod0vsUniform, act_model1_mod1vsUniform, act_Uniform_mod1vsUniform, result_log_name, run_time_log, oppType="Uniform")
  plot_rews(rew_model0_mod0vsUniform, rew_Uniform_mod0vsUniform, rew_model1_mod1vsUniform, rew_Uniform_mod1vsUniform, result_log_name, run_time_log, num_trials, oppType="Uniform")
  display_percentage_of_hand(act_model0_mod0vsTendR, act_TendR_mod0vsTendR, act_model1_mod1vsTendR, act_TendR_mod1vsTendR, result_log_name, run_time_log, oppType="TendR")
  plot_rews(rew_model0_mod0vsTendR, rew_TendR_mod0vsTendR, rew_model1_mod1vsTendR, rew_TendR_mod1vsTendR, result_log_name, run_time_log, num_trials, oppType="TendR")
  display_percentage_of_hand(act_model0_mod0vsTendC, act_TendC_mod0vsTendC, act_model1_mod1vsTendC, act_TendC_mod1vsTendC, result_log_name, run_time_log, oppType="TendC")
  plot_rews(rew_model0_mod0vsTendC, rew_TendC_mod0vsTendC, rew_model1_mod1vsTendC, rew_TendC_mod1vsTendC, result_log_name, run_time_log, num_trials, oppType="TendC")
  display_percentage_of_hand(act_model0_mod0vsTendP, act_TendP_mod0vsTendP, act_model1_mod1vsTendP, act_TendP_mod1vsTendP, result_log_name, run_time_log, oppType="TendP")
  plot_rews(rew_model0_mod0vsTendP, rew_TendP_mod0vsTendP, rew_model1_mod1vsTendP, rew_TendP_mod1vsTendP, result_log_name, run_time_log, num_trials, oppType="TendP")
  display_percentage_of_hand(act_model0_mod0vsR, act_R_mod0vsR, act_model1_mod1vsR, act_R_mod1vsR, result_log_name, run_time_log, oppType="R")
  plot_rews(rew_model0_mod0vsR, rew_R_mod0vsR, rew_model1_mod1vsR, rew_R_mod1vsR, result_log_name, run_time_log, num_trials, oppType="R")
  display_percentage_of_hand(act_model0_mod0vsC, act_C_mod0vsC, act_model1_mod1vsC, act_C_mod1vsC, result_log_name, run_time_log, oppType="C")
  plot_rews(rew_model0_mod0vsC, rew_C_mod0vsC, rew_model1_mod1vsC, rew_C_mod1vsC, result_log_name, run_time_log, num_trials, oppType="C")
  display_percentage_of_hand(act_model0_mod0vsP, act_P_mod0vsP, act_model1_mod1vsP, act_P_mod1vsP, result_log_name, run_time_log, oppType="P")
  plot_rews(rew_model0_mod0vsP, rew_P_mod0vsP, rew_model1_mod1vsP, rew_P_mod1vsP, result_log_name, run_time_log, num_trials, oppType="P")


  all_finish_time = time.time()
  print(f"all finish {(all_finish_time - start_time)/60:.2f} min\n{result_log_name}")

if __name__ == "__main__":
  main()
