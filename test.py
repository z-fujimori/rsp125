# from opp_buffer import DQN
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from rsp125 import RSP125
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_display import plot_rews,display_percentage_of_hand,plot_hand_hist_csv,retaliating_plot_rews,two_lines_rew_plot,robust_evaluation,three_col_robust_evaluation,two_col_robust_evaluation

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


  num_trials = 5
  learn_rate = 0.0008   #  学習率 DQNのデフォルトは1e-3
  learn_rate_leverage = 1.4   #  !!!!!!!!!!!!!!!!!!model0がのんびりさんだ、、、、なぜだ
  gamma = 0.99    #    割引率   デフォルトは0.99
  gradient_steps = 1000 # learn()ごとに何回学習するか デフォルトは１ 
  batch_size = 256 #  default=256
  # gradient_steps × batch_size が1回のトレーニングで使用されるサンプル数
  freq_step = 10
  freq_word = "episode"
  train_freq = (freq_step, freq_word) # 何ステップごとにモデルのトレーニングを行うか default=(1, "step")
  seed_value = 42 # シードを揃える
  policy_kwargs = dict(net_arch=[128, 128, 128]) # ネットワークのアーキテクチャを変更 デフォルトは[64, 64]

  print(f"num_trials{num_trials} learn_rate{learn_rate} learn_rate_leverage{learn_rate_leverage} gamma{gamma} gradient_steps{gradient_steps} batch_size{batch_size} freq_step{freq_step}{freq_word} seed_value{seed_value}")

  act_len_0_timing1 = []
  act_len_1_timing1 = []
  rew_len_0_timing1 = []
  rew_len_1_timing1 = []
  act_len_0_timing2 = []
  act_len_1_timing2 = []
  rew_len_0_timing2 = []
  rew_len_1_timing2 = []
  env0 = RSP125(goal=100, n_history=5)
  env1 = RSP125(goal=100, n_history=5)
  model0 = DQN(
    "MlpPolicy",
    env0,
    seed=seed_value,
    learning_starts=0,  # 元は0 
    train_freq=train_freq,
    gradient_steps=gradient_steps, 
    verbose=0,
    gamma=gamma,
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
    learning_rate=learn_rate,   #  学習りつ
    # device="cuda",
    policy_kwargs=policy_kwargs # nnの設定
  )
  # model0.set_env(RSP125(opp=model1, goal=100))
  # model1.set_env(RSP125(opp=model0, goal=100))
  env0.opp = model1
  env1.opp = model0

  # modelのNNについて
  print("隠れ層,ニューロン数: \n",model1.policy.q_net)

  for i in range(num_trials):
    model0.learn(total_timesteps=1_000, log_interval=100)

    obs, info  = env1.reset()
    for k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, reward, terminated, truncated, info = env1.step(action)
    act_len_0_timing1, rew_len_0_timing1, act_len_1_timing1, rew_len_1_timing1 = append_act_rew_env1(act_len_0_timing1, rew_len_0_timing1, act_len_1_timing1, rew_len_1_timing1, env1._action_history[5:], env1._reward_history)

    model1.learn(total_timesteps=1_000, log_interval=100)

    obs, info  = env0.reset()
    for k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]       # ！！！！！！！！！ここは互い違いでなければいけない？
      obs, reward, terminated, truncated, info = env0.step(action)
    act_len_0_timing2, rew_len_0_timing2, act_len_1_timing2, rew_len_1_timing2 = append_act_rew_env0(act_len_0_timing2, rew_len_0_timing2, act_len_1_timing2, rew_len_1_timing2, env0._action_history[5:], env0._reward_history)
  
    print(f"i: {i}\ntiming1 reward0: {rew_len_0_timing1[i]}, reward1: {rew_len_1_timing1[i]}\ntiming2 reward0: {rew_len_0_timing2[i]}, reward1: {rew_len_1_timing2[i]}")

  end_time = time.time()
  print(f"Execution time: {end_time - start_time:.2f} seconds")

  local_end_time = time.localtime(end_time)
  format_end_time = time.strftime("%Y-%m%d-%H:%M:%S",local_end_time)

  # 保存用ディレクトリ作成
  result_log_name = f"aopp検証_originDQN_mod0*{learn_rate_leverage}_{format_end_time}_learningRate{learn_rate}_gamma{gamma}_gradientSteps{gradient_steps}_trainFreq{freq_step}{freq_word}_trial{num_trials}_batchSize{batch_size}_seed{seed_value}"
  # os.makedirs(f"./results/{result_log_name}", exist_ok=True)
  # os.makedirs(f"./results/{result_log_name}/hand_csv", exist_ok=True)
  # os.makedirs(f"./results/{result_log_name}/rew_plot", exist_ok=True)

  all_plot_time = time.time()
  # run_time_log = f"all plot {(all_plot_time - start_time)/60:.2f} min\n{result_log_name}"
  # display_percentage_of_hand(act_len_0_timing1, act_len_1_timing1, act_len_0_timing2, act_len_1_timing2, result_log_name, run_time_log)
  # plot_rews(rew_len_0_timing1, rew_len_1_timing1, rew_len_0_timing2, rew_len_1_timing2, result_log_name, run_time_log)

  all_finish_time = time.time()
  print(f"all finish {(all_finish_time - start_time)/60:.2f} min\n{result_log_name}")

def hand():
  csv_file_name = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach256_2025-0209-00:07:09_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust/tendR/hand_(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach256_2025-0209-00:07:09_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5_timing1.png"
  save_name = "fig7-エージェントAとOnlyC"

  plot_hand_hist_csv(csv_file_name, save_name)

def plot_rew_from_npy(path,save_name):
  rews1_timing1 = np.load(f"{path}/rews1_timing1.npy")
  rews1_timing2 = np.load(f"{path}/rews1_timing2.npy")
  rews2_timing1 = np.load(f"{path}/rews2_timing1.npy")
  rews2_timing2 = np.load(f"{path}/rews2_timing2.npy")

  step=1
  move_ave=100
  result_name=f"{save_name}_step{step}_ave{move_ave}"
  num_trials = len(rews1_timing1)
  print(num_trials)

  plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2, result_name=result_name, num_trials=num_trials, step=step, is_save_mode=False, move_ave_num=move_ave)

# (修正版)から始まっていないフォルダはtiming２のデータが誤っている可能性あり AとBが逆
def error_correction_plot_rew_from_npy(path,save_name):
  rews1_timing1 = np.load(f"{path}/rews1_timing1.npy")
  rews1_timing2 = np.load(f"{path}/rews2_timing2.npy")
  rews2_timing1 = np.load(f"{path}/rews2_timing1.npy")
  rews2_timing2 = np.load(f"{path}/rews1_timing2.npy")

  step=1
  move_ave=100
  result_name=f"{save_name}_step{step}_ave{move_ave}"
  num_trials = len(rews1_timing1)
  print(num_trials)

  plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2,result_name=result_name, num_trials=num_trials, step=step,is_save_mode=False, move_ave_num=move_ave)

def robust_plot_rew_from_npy(path,save_name,robustType):
  rews1_timing1 = np.load(f"{path}/rews0_mod0.npy")
  rews1_timing2 = np.load(f"{path}/rews1_mod1.npy")
  rews2_timing1 = np.load(f"{path}/rews{robustType}_mod0.npy")
  rews2_timing2 = np.load(f"{path}/rews{robustType}_mod1.npy")
  print(rews1_timing1)
  print(rews1_timing2)
  print(rews2_timing1)
  print(rews2_timing2)

  step=1
  move_ave=100
  result_name=f"{save_name}_step{step}_ave{move_ave}"
  num_trials = len(rews1_timing1)
  print(num_trials)

  plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2,result_name=result_name, num_trials=num_trials, step=step, is_save_mode=False, move_ave_num=move_ave, oppType="Nash")

def robust_rew():
  robustType = "Nash"
  path = "results/追加検証しっぺ返し追加検証(rsp,nash,uni)_2025-0131-10:42:51_learningRate0.0005_gamma0.99_gradientSteps10_trainFreq10episode_trial1000_batchSize256_seed42/robust/nash"
  save_name = "しっぺ_nash_0131-104251"
  robust_plot_rew_from_npy(path=path, save_name=save_name, robustType=robustType)


def ret_rew_plot():
  path = "results/対しっぺ検証(rsp,nash,uni)_2025-0205-15:20:03_learningRate0.0005_gamma0.99_gradientSteps10_trainFreq10episode_trial1000_batchSize100_seed42/rew_plot"
  save_name = "しっぺ1000_0205-152003"

  rews1_timing1 = np.load(f"{path}/rews1_timing1.npy")
  rews2_timing1 = np.load(f"{path}/rews2_timing1.npy")
  rews1_timing2 = np.load(f"{path}/rews1_timing2.npy")
  rews2_timing2 = np.load(f"{path}/rews2_timing2.npy")
  # rews1_timing1 = np.load(f"{path}/rews0_mod0.npy")
  # rews2_timing1 = np.load(f"{path}/rewsNash_mod0.npy")
  # rews1_timing2 = np.load(f"{path}/rews1_mod1.npy")
  # rews2_timing2 = np.load(f"{path}/rewsNash_mod1.npy")
  step=1
  result_name=f"{save_name}_step{step}"
  num_trials = len(rews1_timing2)
  print(num_trials)

  retaliating_plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2,result_name=result_name, num_trials=num_trials, step=step,is_save_mode=False, )

def two_lines_rew():
  path1 = "results/追加検証(nash,uni)_originDQN_mod0*1.0-gradient*1.0-bach256_2025-0126-07:17:05_learningRate7e-05_gamma0.99_gradientSteps1200_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42/rew_plot"
  path2 = "results/(修正版)追加検証(nash,uni,tend)_originDQN_mod0*1.0-gradient*1.2-bach256_2025-0130-12:46:42_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42/rew_plot"

  rew1_data1 = np.load(f"{path1}/rews1_timing2.npy")
  rew1_data2 = np.load(f"{path1}/rews2_timing2.npy")
  rew1_name1 = "エージェントA"
  rew1_name2 = "エージェントB"
  rew2_data1 = np.load(f"{path2}/rews1_timing2.npy")
  rew2_data2 = np.load(f"{path2}/rews2_timing2.npy")
  rew2_name1 = "エージェントA"
  rew2_name2 = "エージェントB"
  result_name="fig4"

  two_lines_rew_plot(rew1_data1, rew1_data2, rew1_name1, rew1_name2, rew2_data1, rew2_data2, rew2_name1, rew2_name2, result_name=result_name)

def robust():
  # path = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach256_2025-0209-00:07:09_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust"
  # name = "fig7"
  path = "results/追加検証しっぺ返し追加検証(rsp,nash,uni)_2025-0131-10:42:51_learningRate0.0005_gamma0.99_gradientSteps10_trainFreq10episode_trial1000_batchSize256_seed42/robust"
  name = "fig6"

  dqn_nash = np.load(f"{path}/nash/rews1_mod1.npy")
  opp_nash = np.load(f"{path}/nash/rewsNash_mod1.npy")
  dqn_r = np.load(f"{path}/R/rews0_mod0.npy")
  opp_r = np.load(f"{path}/R/rewsUni_mod0.npy")
  dqn_c = np.load(f"{path}/C/rews0_mod0.npy")
  opp_c = np.load(f"{path}/C/rewsUni_mod0.npy")
  dqn_p = np.load(f"{path}/P/rews0_mod0.npy")
  opp_p = np.load(f"{path}/P/rewsUni_mod0.npy")

  robust_evaluation(name, dqn_nash, opp_nash, dqn_r, opp_r, dqn_c, opp_c, dqn_p, opp_p)

def two_robust():
  path = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach256_2025-0209-00:07:09_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust"
  name = "fig7"
  # path = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach256_2025-0209-14:33:05_learningRate0.000126_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust"
  # name = "fig_lr_highhigh"
  # path = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach400_2025-0209-17:56:54_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize400_nn[64, 64]_seed42_history5/robust"
  # name = "fig_bs_highhigh"
  # path = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach256_2025-0209-18:06:59_learningRate7e-05_gamma0.99_gradientSteps1200_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust"
  # name = "fig_gs_highhigh"

  dqn_nash_0 = np.load(f"{path}/nash/rews0_mod0.npy")
  opp_nash_0 = np.load(f"{path}/nash/rewsNash_mod0.npy")
  dqn_r_0 = np.load(f"{path}/R/rews0_mod0.npy")
  opp_r_0 = np.load(f"{path}/R/rewsUni_mod0.npy")
  dqn_c_0 = np.load(f"{path}/C/rews0_mod0.npy")
  opp_c_0 = np.load(f"{path}/C/rewsUni_mod0.npy")
  dqn_p_0 = np.load(f"{path}/P/rews0_mod0.npy")
  opp_p_0 = np.load(f"{path}/P/rewsUni_mod0.npy")
  dqn_nash_1 = np.load(f"{path}/nash/rews1_mod1.npy")
  opp_nash_1 = np.load(f"{path}/nash/rewsNash_mod1.npy")
  dqn_r_1 = np.load(f"{path}/R/rews1_mod1.npy")
  opp_r_1 = np.load(f"{path}/R/rewsUni_mod1.npy")
  dqn_c_1 = np.load(f"{path}/C/rews1_mod1.npy")
  opp_c_1 = np.load(f"{path}/C/rewsUni_mod1.npy")
  dqn_p_1 = np.load(f"{path}/P/rews1_mod1.npy")
  opp_p_1 = np.load(f"{path}/P/rewsUni_mod1.npy")

  two_col_robust_evaluation(name, dqn_nash_0, opp_nash_0, dqn_r_0, opp_r_0, dqn_c_0, opp_c_0, dqn_p_0, opp_p_0, dqn_nash_1, opp_nash_1, dqn_r_1, opp_r_1, dqn_c_1, opp_c_1, dqn_p_1, opp_p_1)

def three_robust():
  # path = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach256_2025-0209-14:33:05_learningRate0.000126_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust"
  # high_low_path = "results/(rsp,nash,uni,tend)_mod01.8-gradient1.0-bach256_2025-0210-202919_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust"
  # name = "fig8"
  path = "results/(rsp,nash,uni,tend)_mod0*1.0-gradient*1.0-bach400_2025-0209-17:56:54_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize400_nn[64, 64]_seed42_history5/robust"
  high_low_path = "results/(rsp,nash,uni,tend)_mod01.8-gradient1.0-bach256_2025-0210-202919_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_history5/robust"
  name = "fig10"

  dqn_nash = np.load(f"{path}/nash/rews1_mod1.npy")
  opp_nash = np.load(f"{path}/nash/rewsNash_mod1.npy")
  dqn_r = np.load(f"{path}/R/rews0_mod0.npy")
  opp_r = np.load(f"{path}/R/rewsUni_mod0.npy")
  dqn_c = np.load(f"{path}/C/rews0_mod0.npy")
  opp_c = np.load(f"{path}/C/rewsUni_mod0.npy")
  dqn_p = np.load(f"{path}/P/rews0_mod0.npy")
  opp_p = np.load(f"{path}/P/rewsUni_mod0.npy")
  high_dqn_nash = np.load(f"{high_low_path}/nash/rews0_mod0.npy")
  high_opp_nash = np.load(f"{high_low_path}/nash/rewsNash_mod0.npy")
  high_dqn_r = np.load(f"{high_low_path}/R/rews0_mod0.npy")
  high_opp_r = np.load(f"{high_low_path}/R/rewsUni_mod0.npy")
  high_dqn_c = np.load(f"{high_low_path}/C/rews0_mod0.npy")
  high_opp_c = np.load(f"{high_low_path}/C/rewsUni_mod0.npy")
  high_dqn_p = np.load(f"{high_low_path}/P/rews0_mod0.npy")
  high_opp_p = np.load(f"{high_low_path}/P/rewsUni_mod0.npy")
  low_dqn_nash = np.load(f"{high_low_path}/nash/rews1_mod1.npy")
  low_opp_nash = np.load(f"{high_low_path}/nash/rewsNash_mod1.npy")
  low_dqn_r = np.load(f"{high_low_path}/R/rews1_mod1.npy")
  low_opp_r = np.load(f"{high_low_path}/R/rewsUni_mod1.npy")
  low_dqn_c = np.load(f"{high_low_path}/C/rews1_mod1.npy")
  low_opp_c = np.load(f"{high_low_path}/C/rewsUni_mod1.npy")
  low_dqn_p = np.load(f"{high_low_path}/P/rews1_mod1.npy")
  low_opp_p = np.load(f"{high_low_path}/P/rewsUni_mod1.npy")

  three_col_robust_evaluation(name, dqn_nash, opp_nash, dqn_r, opp_r, dqn_c, opp_c, dqn_p, opp_p, high_dqn_nash, high_opp_nash, high_dqn_r, high_opp_r, high_dqn_c, high_opp_c, high_dqn_p, high_opp_p, low_dqn_nash, low_opp_nash, low_dqn_r, low_opp_r, low_dqn_c, low_opp_c, low_dqn_p, low_opp_p)


if __name__ == "__main__":
  # main()
  # hand()
  # ret_rew_plot()
  # robust_rew()
  # two_lines_rew()
  robust()
  two_robust()
  # three_robust()

  # read_npy = np.load(f"results/追加検証(nash,uni)_originDQN_mod0*1.0-gradient*1.0-bach256_2025-0125-03:54:04_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42/robust/nash/rewsNash_mod1.npy")
  # print(read_npy)
  # print(np.mean(read_npy))

  # path = "results/追加検証(nash,uni)_originDQN_mod01.0-gradient1.0-bach256_2025-0125-201150_learningRate7e-05_gamma0.98_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42/rew_plot"
  # plot_rew_from_npy(path,"0125-20:11:50_gamma0.98")
  # error_correction_plot_rew_from_npy(path,"(修正版)0125-071705_gradient1200")

