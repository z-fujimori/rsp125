from stable_baselines3 import DQN
from rsp125 import RSP125
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def append_act_rew_env0(act_0, rew_0, act_1, rew_1, act_hist, rew_hist):
  for a in act_hist:
    act_0.append([a[0]])  
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
    act_0.append([a[1]])  
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
  learn_rate = 0.00007    # 学習率
  learn_rate_leverage = 1.0    # Aの学習率 = B * leverrage
  gamma = 0.99    # 割引率 
  gradient_steps = 1000    
  gradient_steps_skale = 1.2    # Aのgradient_steps = B * skale
  batch_size = 256
  model0_batch_size = 256
  freq_step = 10
  freq_word = "episode"
  train_freq = (freq_step, freq_word)
  layer = [64,64]
  policy_kwargs = dict(net_arch=layer)
  seed_value = 42
  save_model_zip = True
  n_history = 5

  act_len_0_timing1 = []
  act_len_1_timing1 = []
  rew_len_0_timing1 = []
  rew_len_1_timing1 = []
  act_len_0_timing2 = []
  act_len_1_timing2 = []
  rew_len_0_timing2 = []
  rew_len_1_timing2 = []
  # robust検証
  # Nash
  act_model0_mod0vsNash = []
  act_Nash_mod0vsNash = []
  rew_model0_mod0vsNash = []
  rew_Nash_mod0vsNash = []
  act_model1_mod1vsNash = []
  act_Nash_mod1vsNash = []
  rew_model1_mod1vsNash = []
  rew_Nash_mod1vsNash = []
  # OnlyR
  act_model0_mod0vsR = []    
  act_R_mod0vsR = []
  rew_model0_mod0vsR = []
  rew_R_mod0vsR = []
  act_model1_mod1vsR = []
  act_R_mod1vsR = []
  rew_model1_mod1vsR = []
  rew_R_mod1vsR = []
  # OnlyC
  act_model0_mod0vsC = []    
  act_C_mod0vsC = []
  rew_model0_mod0vsC = []
  rew_C_mod0vsC = []
  act_model1_mod1vsC = []
  act_C_mod1vsC = []
  rew_model1_mod1vsC = []
  rew_C_mod1vsC = []
  # OnlyP
  act_model0_mod0vsP = []    
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
    learning_starts=0,  
    train_freq=train_freq,
    gradient_steps=int(gradient_steps*gradient_steps_skale), 
    verbose=0,
    gamma=gamma,
    batch_size=model0_batch_size,
    learning_rate=learn_rate*learn_rate_leverage,  
    policy_kwargs=policy_kwargs 
  )
  model1 = DQN(
    "MlpPolicy", 
    env1, 
    seed=seed_value+1,
    learning_starts=0, 
    train_freq=train_freq,
    gradient_steps=gradient_steps,
    verbose=0,
    gamma=gamma,
    batch_size=batch_size,
    learning_rate=learn_rate,   
    policy_kwargs=policy_kwargs 
  )
  env0.opp = model1
  env1.opp = model0
  # robust検証
  envNash = RSP125(goal=100, n_history=n_history, oppType="Nash")
  envR = RSP125(goal=100, n_history=n_history, oppType="R")
  envC = RSP125(goal=100, n_history=n_history, oppType="C")
  envP = RSP125(goal=100, n_history=n_history, oppType="P")

  for _i in range(num_trials):
    # model0学習
    model0.learn(total_timesteps=1_000, log_interval=100)

    # model0に行動させた評価
    obs, _info  = env0.reset()
    for _k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]   
      obs, _reward, _terminated, _truncated, _info = env0.step(action)
    act_len_0_timing2, rew_len_0_timing2, act_len_1_timing2, rew_len_1_timing2 = append_act_rew_env0(act_len_0_timing2, rew_len_0_timing2, act_len_1_timing2, rew_len_1_timing2, env0._action_history[5:], env0._reward_history)
    # robust検証
    obs, _info = envNash.reset()
    for _k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envNash.step(action)
    act_model0_mod0vsNash, rew_model0_mod0vsNash, act_Nash_mod0vsNash, rew_Nash_mod0vsNash = append_act_rew_env0(act_model0_mod0vsNash, rew_model0_mod0vsNash, act_Nash_mod0vsNash, rew_Nash_mod0vsNash, envNash._action_history[5:], envNash._reward_history)
    obs, _info = envR.reset()
    for _k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envR.step(action)
    act_model0_mod0vsR, rew_model0_mod0vsR, act_R_mod0vsR, rew_R_mod0vsR = append_act_rew_env0(act_model0_mod0vsR, rew_model0_mod0vsR, act_R_mod0vsR, rew_R_mod0vsR, envR._action_history[5:], envR._reward_history)
    obs, _info = envC.reset()
    for _k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envC.step(action)
    act_model0_mod0vsC, rew_model0_mod0vsC, act_C_mod0vsC, rew_C_mod0vsC = append_act_rew_env0(act_model0_mod0vsC, rew_model0_mod0vsC, act_C_mod0vsC, rew_C_mod0vsC, envC._action_history[5:], envC._reward_history)
    obs, _info = envP.reset()
    for _k in range(goal):
      action = model0.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envP.step(action)
    act_model0_mod0vsP, rew_model0_mod0vsP, act_P_mod0vsP, rew_P_mod0vsP = append_act_rew_env0(act_model0_mod0vsP, rew_model0_mod0vsP, act_P_mod0vsP, rew_P_mod0vsP, envP._action_history[5:], envP._reward_history)

    # model1学習
    model1.learn(total_timesteps=1_000, log_interval=100)

    # model1に行動させた評価
    obs, _info  = env1.reset()
    for _k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = env1.step(action)
    act_len_0_timing1, rew_len_0_timing1, act_len_1_timing1, rew_len_1_timing1 = append_act_rew_env1(act_len_0_timing1, rew_len_0_timing1, act_len_1_timing1, rew_len_1_timing1, env1._action_history[5:], env1._reward_history)
    # robust検証
    obs, _info = envNash.reset()
    for _k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envNash.step(action)
    act_model1_mod1vsNash, rew_model1_mod1vsNash, act_Nash_mod1vsNash, rew_Nash_mod1vsNash = append_act_rew_env0(act_model1_mod1vsNash, rew_model1_mod1vsNash, act_Nash_mod1vsNash, rew_Nash_mod1vsNash, envNash._action_history[5:], envNash._reward_history)
    obs, _info = envR.reset()
    for _k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envR.step(action)
    act_model1_mod1vsR, rew_model1_mod1vsR, act_R_mod1vsR, rew_R_mod1vsR = append_act_rew_env0(act_model1_mod1vsR, rew_model1_mod1vsR, act_R_mod1vsR, rew_R_mod1vsR, envR._action_history[5:], envR._reward_history)
    obs, _info = envC.reset()
    for _k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envC.step(action)
    act_model1_mod1vsC, rew_model1_mod1vsC, act_C_mod1vsC, rew_C_mod1vsC = append_act_rew_env0(act_model1_mod1vsC, rew_model1_mod1vsC, act_C_mod1vsC, rew_C_mod1vsC, envC._action_history[5:], envC._reward_history)
    obs, _info = envP.reset()
    for _k in range(goal):
      action = model1.predict(obs, deterministic=True)[0]
      obs, _reward, _terminated, _truncated, _info = envP.step(action)
    act_model1_mod1vsP, rew_model1_mod1vsP, act_P_mod1vsP, rew_P_mod1vsP = append_act_rew_env0(act_model1_mod1vsP, rew_model1_mod1vsP, act_P_mod1vsP, rew_P_mod1vsP, envP._action_history[5:], envP._reward_history)

  # 保存用ディレクトリ作成
  result_name = f"log_name"
  os.makedirs(f"./results/{result_name}", exist_ok=True)
  os.makedirs(f"./results/{result_name}/hand_csv", exist_ok=True)
  os.makedirs(f"./results/{result_name}/rew_plot", exist_ok=True)
  os.makedirs(f"./results/{result_name}/robust/nash", exist_ok=True)
  os.makedirs(f"./results/{result_name}/robust/R", exist_ok=True)
  os.makedirs(f"./results/{result_name}/robust/C", exist_ok=True)
  os.makedirs(f"./results/{result_name}/robust/P", exist_ok=True)

  if save_model_zip:
    os.makedirs(f"./model_zips/{result_name}", exist_ok=True)
    model0.save(f"./model_zips/{result_name}/model0.zip")
    model1.save(f"./model_zips/{result_name}/model1.zip")

  plot_rews(rew_len_0_timing1, rew_len_1_timing1, rew_len_0_timing2, rew_len_1_timing2, result_name, num_trials)
  # robust検証
  plot_rews(rew_model0_mod0vsNash, rew_Nash_mod0vsNash, rew_model1_mod1vsNash, rew_Nash_mod1vsNash, result_name, num_trials, oppType="Nash")
  plot_rews(rew_model0_mod0vsR, rew_R_mod0vsR, rew_model1_mod1vsR, rew_R_mod1vsR, result_name, num_trials, oppType="R")
  plot_rews(rew_model0_mod0vsC, rew_C_mod0vsC, rew_model1_mod1vsC, rew_C_mod1vsC, result_name, num_trials, oppType="C")
  plot_rews(rew_model0_mod0vsP, rew_P_mod0vsP, rew_model1_mod1vsP, rew_P_mod1vsP, result_name, num_trials, oppType="P")

def plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2, result_name='output', num_trials=10000, step=1, move_ave_num=100, oppType=None):
  if oppType == "Nash":
    np.save(f"./results/{result_name}/robust/nash/rews0_mod0",rews1_timing1)
    np.save(f"./results/{result_name}/robust/nash/rewsNash_mod0",rews2_timing1)
    np.save(f"./results/{result_name}/robust/nash/rews1_mod1",rews1_timing2)
    np.save(f"./results/{result_name}/robust/nash/rewsNash_mod1",rews2_timing2)
  elif oppType == "R":
    np.save(f"./results/{result_name}/robust/R/rews0_mod0",rews1_timing1)
    np.save(f"./results/{result_name}/robust/R/rewsUni_mod0",rews2_timing1)
    np.save(f"./results/{result_name}/robust/R/rews1_mod1",rews1_timing2)
    np.save(f"./results/{result_name}/robust/R/rewsUni_mod1",rews2_timing2)
  elif oppType == "C":
    np.save(f"./results/{result_name}/robust/C/rews0_mod0",rews1_timing1)
    np.save(f"./results/{result_name}/robust/C/rewsUni_mod0",rews2_timing1)
    np.save(f"./results/{result_name}/robust/C/rews1_mod1",rews1_timing2)
    np.save(f"./results/{result_name}/robust/C/rewsUni_mod1",rews2_timing2)
  elif oppType == "P":
    np.save(f"./results/{result_name}/robust/P/rews0_mod0",rews1_timing1)
    np.save(f"./results/{result_name}/robust/P/rewsUni_mod0",rews2_timing1)
    np.save(f"./results/{result_name}/robust/P/rews1_mod1",rews1_timing2)
    np.save(f"./results/{result_name}/robust/P/rewsUni_mod1",rews2_timing2)
  else:
    np.save(f"./results/{result_name}/rew_plot/rews1_timing1",rews1_timing1)
    np.save(f"./results/{result_name}/rew_plot/rews2_timing1",rews2_timing1)
    np.save(f"./results/{result_name}/rew_plot/rews1_timing2",rews1_timing2)
    np.save(f"./results/{result_name}/rew_plot/rews2_timing2",rews2_timing2)
  
  def save_plot_rews(rews1, rews2, log_dir, log_name):
    # xとrewsの長さを調整しながら間引き
    min_len = min(len(rews1), len(rews2)) // step
    x = np.arange(1, min_len + 1)
    rews1 = rews1[:min_len * step:step]
    rews2 = rews2[:min_len * step:step]
    plt.rcParams['font.family'] = 'Osaka'
    mpl.rcParams.update({'font.size': 150})
    # メモリの文字サイズを小さくする
    plt.tick_params(axis='both', labelsize=100)
    # プロット
    plt.figure(figsize=(65, 30))  # グラフのサイズを指定
    if oppType == "Nash":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="Nash", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "R":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="R", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "C":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="C", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "P":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="P", color="orange", alpha=0.6, linewidth=12)
    else:
      plt.plot(x, rews1, label="PlayerA", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="PlayerB", color="orange", alpha=0.6, linewidth=12)
    # 横軸ラベルを変更するためのFormatter
    def multiply_by_five(x, pos):
      return f'{x * step:.0f}'  # 倍率step倍にし、小数点なしのフォーマット
    # グラフの設定
    # 軸設定
    plt.gca().xaxis.set_major_formatter(FuncFormatter(multiply_by_five))
    plt.xlim(0, num_trials//step)
    plt.ylim(1,340)
    plt.xticks(rotation=90) # 横軸のメモリの表記を縦書きにする
    plt.title(f'{log_name}', fontsize=32)
    plt.xlabel("trial")
    plt.ylabel("合計得点")
    plt.legend(framealpha=0.7)  # 凡例を追加
    # y=250 の水平線を引く
    plt.axhline(y=250, color='r', linestyle='--', linewidth=8)
    # メジャー目盛り
    plt.gca().xaxis.set_major_locator(MultipleLocator(1000))
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    # マイナー目盛り
    plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(25))
    plt.minorticks_on()    # マイナー目盛りを有効化
    # グリッド表示
    plt.grid(which='minor', linestyle=':', linewidth=2)
    plt.grid(which='major', linestyle='-', linewidth=4)
    plt.tight_layout()  # レイアウトを調整
    # 凡例の線の太さ
    leg = plt.legend()
    leg.get_lines()[0].set_linewidth(20)
    leg.get_lines()[1].set_linewidth(20)
    # 保存
    match oppType:
      case "Nash": file_path = os.path.join(f"./results/{log_dir}/robust/nash", f"{log_name}.png")
      case "R": file_path = os.path.join(f"./results/{log_dir}/robust/R", f"{log_name}.png") 
      case "C": file_path = os.path.join(f"./results/{log_dir}/robust/C", f"{log_name}.png") 
      case "P": file_path = os.path.join(f"./results/{log_dir}/robust/P", f"{log_name}.png") 
      case _: file_path = os.path.join(f"./results/{log_dir}/rew_plot", f"{log_name}.png") 
    plt.savefig(file_path)

  new_rews1_timing1 = []
  new_rews2_timing1 = []
  new_rews1_timing2 = []
  new_rews2_timing2 = []
  for i in range(len(rews1_timing1)-move_ave_num):
    new_rews1_timing1.append(sum(rews1_timing1[i:i+move_ave_num])/move_ave_num)
    new_rews2_timing1.append(sum(rews2_timing1[i:i+move_ave_num])/move_ave_num)
    new_rews1_timing2.append(sum(rews1_timing2[i:i+move_ave_num])/move_ave_num)
    new_rews2_timing2.append(sum(rews2_timing2[i:i+move_ave_num])/move_ave_num)
  rews1_timing1 = new_rews1_timing1
  rews2_timing1 = new_rews2_timing1
  rews1_timing2 = new_rews1_timing2
  rews2_timing2 = new_rews2_timing2
  save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"{result_name}_timing1")
  save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"{result_name}_timing2")

if __name__ == "__main__":
  main()
