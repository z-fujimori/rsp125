# from opp_buffer import DQN
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from rsp125 import RSP125
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_display import plot_rews,display_percentage_of_hand,plot_hand_hist_csv,retaliating_plot_rews

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
  csv_file_name = "results/追加検証(nash,uni)_originDQN_mod0*1.8-gradient*1.0-bach256_2025-0125-03:56:38_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42/hand_csv/hand_追加検証(nash,uni)_originDQN_mod0*1.8-gradient*1.0-bach256_2025-0125-03:56:38_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42_timing2.csv"
  save_name = "0125-03:56:38_学習率1.8レバ"

  plot_hand_hist_csv(csv_file_name, save_name)

def plot_rew_from_npy(path,save_name):
  rews1_timing1 = np.load(f"{path}/rews1_timing1.npy")
  rews1_timing2 = np.load(f"{path}/rews1_timing2.npy")
  rews2_timing1 = np.load(f"{path}/rews2_timing1.npy")
  rews2_timing2 = np.load(f"{path}/rews2_timing2.npy")

  step=1
  move_ave=20
  result_name=f"{save_name}_step{step}_ave{move_ave}"
  num_trials = len(rews1_timing1)
  print(num_trials)

  plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2,result_name=result_name, num_trials=num_trials, step=step,is_save_mode=False, move_ave_num=move_ave)

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

  step=1
  move_ave=100
  result_name=f"{save_name}_step{step}_ave{move_ave}"
  num_trials = len(rews1_timing1)
  print(num_trials)

  plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2,result_name=result_name, num_trials=num_trials, step=step, is_save_mode=False, move_ave_num=move_ave, oppType="Uniform")

def robust_rew():
  robustType = "Nash"
  path = "results/追加検証(nash,uni)_originDQN_mod0*1.0-gradient*1.0-bach256_2025-0126-07:17:05_learningRate7e-05_gamma0.99_gradientSteps1200_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42/robust/uniform"
  save_name = "Uni_gradient1200両方_0126-071705"
  robust_plot_rew_from_npy(path=path, save_name=save_name, robustType=robustType)


def ret_rew_plot():
  path = "results/追加検証しっぺ返し_2025-0126-14:44:14_learningRate0.0005_gamma0.99_gradientSteps10_trainFreq10episode_trial1000_batchSize256_seed42/rew_plot"
  save_name = "しっぺ返し1000_0126-144414"

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


if __name__ == "__main__":
  # main()
  hand()
  # ret_rew_plot()
  # robust_rew()

  # path = "results/追加検証(nash,uni)_originDQN_mod0*1.0-gradient*1.0-bach400_2025-0125-04:33:01_learningRate7e-05_gamma0.99_gradientSteps1000_trainFreq10episode_trial10000_batchSize256_nn[64, 64]_seed42/rew_plot"
  # plot_rew_from_npy(path,"0125-04:33:01_batch400")
  # error_correction_plot_rew_from_npy(path,"(修正版)0125-071705_gradient1200")
