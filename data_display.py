import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
import csv
import os

# フォントを日本語対応のものに設定
rcParams['font.family'] = 'Osaka'

move_ave = 50 # 移動平均

def plot_rews(rews1_timing1,rews2_timing1,rews1_timing2,rews2_timing2,result_name='output',run_time_log='--',num_trials=10000,step=5,is_save_mode=True, moving_average=True, move_ave_num=10, oppType=None):
  move_ave = move_ave_num
  if (is_save_mode):
    if oppType == "Nash":
      np.save(f"./results/{result_name}/robust/nash/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/nash/rewsNash_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/nash/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/nash/rewsNash_mod1",rews2_timing2)
    elif oppType == "Uniform":
      np.save(f"./results/{result_name}/robust/uniform/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/uniform/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/uniform/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/uniform/rewsUni_mod1",rews2_timing2)
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
    sum_rews = (np.array(rews1) + np.array(rews2)) / 2

    mpl.rcParams.update({'font.size': 150})
    # メモリの文字サイズを小さくする
    plt.tick_params(axis='both', labelsize=100) 

    # プロット
    plt.figure(figsize=(65, 30))  # グラフのサイズを指定
    # plt.plot(x, sum_rews, label="平均点", color="black", alpha=0.5, linewidth=12)
    if oppType == "Nash":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="Nash", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "Uniform":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="Uniform(1/3)", color="orange", alpha=0.6, linewidth=12)
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
    plt.title(f'{log_name}_{run_time_log}min', fontsize=32)
    plt.xlabel("trial")
    plt.ylabel("合計得点")
    plt.legend(framealpha=0.7)  # 凡例を追加
    # y=250 の水平線を引く
    plt.axhline(y=250, color='r', linestyle='--', linewidth=8)
    # plt.axhline(y=250, color='r', linestyle='--', linewidth=5, label='y = 250')
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
    # leg.get_lines()[2].set_linewidth(20)
    # 保存
    if (is_save_mode):
      if oppType == "Nash":
        file_path = os.path.join(f"./results/{log_dir}/robust/nash", f"{log_name}.png") 
      elif oppType == "Uniform":
        file_path = os.path.join(f"./results/{log_dir}/robust/uniform", f"{log_name}.png") 
      else:
        file_path = os.path.join(f"./results/{log_dir}/rew_plot", f"{log_name}.png") 
    else:
      os.makedirs(f"./make_results/{log_dir}", exist_ok=True)
      os.makedirs(f"./make_results/{log_dir}/rew_plot", exist_ok=True)
      file_path = os.path.join(f"./make_results/{log_dir}/rew_plot", f"{log_name}.png")

    plt.savefig(file_path)

  if moving_average :
    new_rews1_timing1 = []
    new_rews2_timing1 = []
    new_rews1_timing2 = []
    new_rews2_timing2 = []

    for i in range(len(rews1_timing1)-move_ave):
      new_rews1_timing1.append(sum(rews1_timing1[i:i+move_ave])/move_ave)
      new_rews2_timing1.append(sum(rews2_timing1[i:i+move_ave])/move_ave)
      new_rews1_timing2.append(sum(rews1_timing2[i:i+move_ave])/move_ave)
      new_rews2_timing2.append(sum(rews2_timing2[i:i+move_ave])/move_ave)

    rews1_timing1 = new_rews1_timing1
    rews2_timing1 = new_rews2_timing1
    rews1_timing2 = new_rews1_timing2
    rews2_timing2 = new_rews2_timing2

  if oppType == "Nash":
    save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"rew_{result_name}_エージェントA")
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_{result_name}_エージェントB")
  elif oppType == "Uniform":
    save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"rew_{result_name}_エージェントA")
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_{result_name}_エージェントB")
  else:
    save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"{result_name}_timing1")
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"{result_name}_timing2")


def display_percentage_of_hand(hist_a_timing1, hist_b_timing1, hist_a_timing2, hist_b_timing2, result_name='output',run_time_log='--', moving_averae=True, oppType=None):

  def save_hand_hist(hist_a, hist_b, log_dir, log_name):
    percentage = np.zeros((3, 3), dtype=int) 
    total_pers = []

    for i, (hand_a, hand_b) in enumerate(zip(hist_a, hist_b)):
      hand_a = int(hand_a[0])
      hand_b = int(hand_b[0])
      percentage[hand_a][hand_b] += 1
      if i%100 == 99:
        total_pers.append(percentage)
        percentage = np.zeros((3, 3), dtype=int)

    # CSV保存部分
    if oppType == "Nash":
      result_name = f"./results/{log_dir}/robust/nash/{log_name}.csv"
    elif oppType == "Uniform":
      result_name = f"./results/{log_dir}/robust/uniform/{log_name}.csv"
    else:
      result_name = f"./results/{log_dir}/hand_csv/{log_name}.csv"

    with open(result_name, mode='w', newline='') as file:
      writer = csv.writer(file)
      # ヘッダーを書き込む
      writer.writerow(["Round", "a\\b", "G", "C", "P","","run time", f"{run_time_log} min"])
      for i, per in enumerate(total_pers):
        # ラウンド名
        writer.writerow([f'Round {i+1}'])
        # データ行を書き込む
        writer.writerow(["", "G", *per[0]])
        writer.writerow(["", "C", *per[1]])
        writer.writerow(["", "P", *per[2]])
        writer.writerow([])  # 空行を挿入して次のラウンドと区切る
  
    # plotデータ作成
    hand_plot_hist = [
      [[],[],[]],
      [[],[],[]],
      [[],[],[]]
    ] 
    for i in range(int(len(total_pers)/3)):
      hand_plot_hist[0][0].append(total_pers[i*3][0])
      hand_plot_hist[0][1].append(total_pers[i*3][1])
      hand_plot_hist[0][2].append(total_pers[i*3][2])
      hand_plot_hist[1][0].append(total_pers[i*3+1][0])
      hand_plot_hist[1][1].append(total_pers[i*3+1][1])
      hand_plot_hist[1][2].append(total_pers[i*3+1][2])
      hand_plot_hist[2][0].append(total_pers[i*3+2][0])
      hand_plot_hist[2][1].append(total_pers[i*3+2][1])
      hand_plot_hist[2][2].append(total_pers[i*3+2][2])

    if moving_averae:
      new_hand00 = []
      new_hand01 = []
      new_hand02 = []
      new_hand10 = []
      new_hand11 = []
      new_hand12 = []
      new_hand20 = []
      new_hand21 = []
      new_hand22 = []

      for i in range(len(hand_plot_hist[0][0])-move_ave):
        new_hand00.append(sum(hand_plot_hist[0][0][i:i+move_ave])/move_ave)
        new_hand01.append(sum(hand_plot_hist[0][1][i:i+move_ave])/move_ave)
        new_hand02.append(sum(hand_plot_hist[0][2][i:i+move_ave])/move_ave)
        new_hand10.append(sum(hand_plot_hist[1][0][i:i+move_ave])/move_ave)
        new_hand11.append(sum(hand_plot_hist[1][1][i:i+move_ave])/move_ave)
        new_hand12.append(sum(hand_plot_hist[1][2][i:i+move_ave])/move_ave)
        new_hand20.append(sum(hand_plot_hist[2][0][i:i+move_ave])/move_ave)
        new_hand21.append(sum(hand_plot_hist[2][1][i:i+move_ave])/move_ave)
        new_hand22.append(sum(hand_plot_hist[2][2][i:i+move_ave])/move_ave)

      hand_plot_hist[0][0] = new_hand00
      hand_plot_hist[0][1] = new_hand01
      hand_plot_hist[0][2] = new_hand02
      hand_plot_hist[1][0] = new_hand10
      hand_plot_hist[1][1] = new_hand11
      hand_plot_hist[1][2] = new_hand12
      hand_plot_hist[2][0] = new_hand20
      hand_plot_hist[2][1] = new_hand21
      hand_plot_hist[2][2] = new_hand22

    hand_plt = create_plt_data(hand_plot_hist[0][0],hand_plot_hist[0][1],hand_plot_hist[0][2],hand_plot_hist[1][0],hand_plot_hist[1][1],hand_plot_hist[1][2],hand_plot_hist[2][0],hand_plot_hist[2][1],hand_plot_hist[2][2])
    if oppType == "Nash":
      file_path = os.path.join(f"./results/{log_dir}/robust/nash", f"{log_name}.png")
    elif oppType == "Uniform":
      file_path = os.path.join(f"./results/{log_dir}/robust/uniform", f"{log_name}.png")
    else:
      file_path = os.path.join(f"./results/{log_dir}/hand_csv", f"{log_name}.png")
    hand_plt.savefig(file_path)

  save_hand_hist(hist_a_timing1, hist_b_timing1, result_name, f"hand_{result_name}_timing1")
  save_hand_hist(hist_a_timing2, hist_b_timing2, result_name, f"hand_{result_name}_timing2")


def plot_hand_hist_csv(csv_file, savename, moving_averae=True):
  df = pd.read_csv(csv_file)
  df = df.drop(0)
  # ROUND 〜　を削除
  indices_to_remove = range(4, len(df), 4)
  df = df.drop(indices_to_remove)
  
  trial = len(df)/3
  hand_plot_hist = [
    [[],[],[]],
    [[],[],[]],
    [[],[],[]]
  ] 
  for i in range(int(trial)):
    hand_plot_hist[0][0].append(df.iloc[i*3]["G"])
    hand_plot_hist[0][1].append(df.iloc[i*3]["C"])
    hand_plot_hist[0][2].append(df.iloc[i*3]["P"])
    hand_plot_hist[1][0].append(df.iloc[i*3+1]["G"])
    hand_plot_hist[1][1].append(df.iloc[i*3+1]["C"])
    hand_plot_hist[1][2].append(df.iloc[i*3+1]["P"])
    hand_plot_hist[2][0].append(df.iloc[i*3+2]["G"])
    hand_plot_hist[2][1].append(df.iloc[i*3+2]["C"])
    hand_plot_hist[2][2].append(df.iloc[i*3+2]["P"])

  if moving_averae:
    new_hand00 = []
    new_hand01 = []
    new_hand02 = []
    new_hand10 = []
    new_hand11 = []
    new_hand12 = []
    new_hand20 = []
    new_hand21 = []
    new_hand22 = []
    for i in range(len(hand_plot_hist[0][0])-move_ave):
      new_hand00.append(sum(hand_plot_hist[0][0][i:i+move_ave])/move_ave)
      new_hand01.append(sum(hand_plot_hist[0][1][i:i+move_ave])/move_ave)
      new_hand02.append(sum(hand_plot_hist[0][2][i:i+move_ave])/move_ave)
      new_hand10.append(sum(hand_plot_hist[1][0][i:i+move_ave])/move_ave)
      new_hand11.append(sum(hand_plot_hist[1][1][i:i+move_ave])/move_ave)
      new_hand12.append(sum(hand_plot_hist[1][2][i:i+move_ave])/move_ave)
      new_hand20.append(sum(hand_plot_hist[2][0][i:i+move_ave])/move_ave)
      new_hand21.append(sum(hand_plot_hist[2][1][i:i+move_ave])/move_ave)
      new_hand22.append(sum(hand_plot_hist[2][2][i:i+move_ave])/move_ave)
    hand_plot_hist[0][0] = new_hand00
    hand_plot_hist[0][1] = new_hand01
    hand_plot_hist[0][2] = new_hand02
    hand_plot_hist[1][0] = new_hand10
    hand_plot_hist[1][1] = new_hand11
    hand_plot_hist[1][2] = new_hand12
    hand_plot_hist[2][0] = new_hand20
    hand_plot_hist[2][1] = new_hand21
    hand_plot_hist[2][2] = new_hand22
    print(len(hand_plot_hist[0][0]))
  
  plt = create_plt_data(hand_plot_hist[0][0],hand_plot_hist[0][1],hand_plot_hist[0][2],hand_plot_hist[1][0],hand_plot_hist[1][1],hand_plot_hist[1][2],hand_plot_hist[2][0],hand_plot_hist[2][1],hand_plot_hist[2][2], file_name=savename)

  plt.savefig(f"./make_hands/{savename}.png")

def create_plt_data(rew_g_g, rew_g_c, rew_g_p, rew_c_g, rew_c_c, rew_c_p, rew_p_g, rew_p_c, rew_p_p, file_name="hist"):
  mpl.rcParams.update({'font.size': 108}) # 文字サイズ
  plt.figure(figsize=(70, 30))  # グラフのサイズを指定
  plt.plot(range(len(rew_g_g)), rew_g_g, label="A: g, B: g", color="brown", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_g_c, label="A: g, B: c", color="orange", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_g_p, label="A: g, B: p", color="blue", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_c_g, label="A: c, B: g", color="green", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_c_c, label="A: c, B: c", color="gray", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_c_p, label="A: c, B: p", color="pink", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_p_g, label="A: p, B: g", color="red", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_p_c, label="A: p, B: c", color="olive", alpha=0.6, linewidth=2.5)
  plt.plot(range(len(rew_g_g)), rew_p_p, label="A: p, B: p", color="purple", alpha=0.6, linewidth=2.5)

  # グラフの設定
  plt.title(f'{file_name}', fontsize=32)
  plt.xlabel("trial")
  plt.ylabel("count")
  plt.xticks(rotation=90) # 横軸のメモリの表記を縦書きにする

  # plt.legend(framealpha=0.7)  # 凡例を追加
  # x=250に太線を引く
  # plt.axvline(y=250, color='red', linewidth=2.5, linestyle='--')
  plt.xlim(-10, len(rew_g_g)+100)
  plt.ylim(-1,99)

  # メジャー目盛り
  plt.gca().xaxis.set_major_locator(MultipleLocator(500))
  plt.gca().yaxis.set_major_locator(MultipleLocator(10))
  # マイナー目盛り
  plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
  plt.gca().yaxis.set_minor_locator(MultipleLocator(5))
  plt.minorticks_on()    # マイナー目盛りを有効化
  # グリッド表示
  plt.grid(which='minor', linestyle=':', linewidth=2)
  plt.grid(which='major', linestyle='-', linewidth=4)
  
  plt.tight_layout()  # レイアウトを調整
  
  # 線の説明の太さ　凡例線の太さ
  leg = plt.legend(fontsize=70, framealpha=0.6) # 凡例文字サイズ, 透過度
  leg.get_lines()[0].set_linewidth(10)
  leg.get_lines()[1].set_linewidth(10)
  leg.get_lines()[2].set_linewidth(10)
  leg.get_lines()[3].set_linewidth(10)
  leg.get_lines()[4].set_linewidth(10)
  leg.get_lines()[5].set_linewidth(10)
  leg.get_lines()[6].set_linewidth(10)
  leg.get_lines()[7].set_linewidth(10)
  leg.get_lines()[8].set_linewidth(10)

  return plt






def retaliating_plot_rews(rews1_timing1,rews2_timing1,rews1_timing2,rews2_timing2,result_name='output',run_time_log='--',num_trials=10000,step=5,is_save_mode=True, moving_average=True, oppType=None):
  if (is_save_mode):
    if oppType == "Nash":
      np.save(f"./results/{result_name}/robust/nash/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/nash/rewsNash_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/nash/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/nash/rewsNash_mod1",rews2_timing2)
    elif oppType == "Uniform":
      np.save(f"./results/{result_name}/robust/uniform/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/uniform/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/uniform/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/uniform/rewsUni_mod1",rews2_timing2)
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
    sum_rews = (np.array(rews1) + np.array(rews2)) / 2

    mpl.rcParams.update({'font.size': 128})

    # プロット
    plt.figure(figsize=(80, 40))  # グラフのサイズを指定
    plt.plot(x, sum_rews, label="平均点", color="black", alpha=0.9, linewidth=12)
    if oppType == "Nash":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.5, linewidth=12)
      plt.plot(x, rews2, label="Nash", color="orange", alpha=0.5, linewidth=12)
    elif oppType == "Uniform":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.5, linewidth=12)
      plt.plot(x, rews2, label="uniform(1/3)", color="orange", alpha=0.5, linewidth=12)
    else:
      plt.plot(x, rews1, label="機械学習エージェント", color="blue", alpha=0.5, linewidth=12)
      plt.plot(x, rews2, label="しっぺ返しエージェント", color="orange", alpha=0.5, linewidth=12)

    # 横軸ラベルを変更するためのFormatter
    def multiply_by_five(x, pos):
      return f'{x * step:.0f}'  # 倍率step倍にし、小数点なしのフォーマット

    # グラフの設定
    # 軸設定
    plt.gca().xaxis.set_major_formatter(FuncFormatter(multiply_by_five))
    # plt.xlim(0, num_trials//step)
    plt.ylim(0,440)
    plt.xticks(rotation=90) # 横軸のメモリの表記を縦書きにする
    plt.title(f'{log_name}_{run_time_log}min', fontsize=32)
    plt.xlabel("trial")
    plt.ylabel("合計得点")
    plt.legend(framealpha=0.7)  # 凡例を追加
    # x=250に太線を引く
    # plt.axvline(y=250, color='red', linewidth=2.5, linestyle='--')
    # メジャー目盛り
    plt.gca().xaxis.set_major_locator(MultipleLocator(250))
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    # マイナー目盛り
    plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(25))
    plt.minorticks_on()    # マイナー目盛りを有効化
    # グリッド表示
    plt.grid(which='minor', linestyle=':', linewidth=2)
    plt.grid(which='major', linestyle='-', linewidth=4)

    plt.tight_layout()  # レイアウトを調整

    leg = plt.legend()

    leg.get_lines()[0].set_linewidth(20)
    leg.get_lines()[1].set_linewidth(20)
    leg.get_lines()[2].set_linewidth(20)
    # 保存
    if (is_save_mode):
      if oppType == "Nash":
        file_path = os.path.join(f"./results/{log_dir}/robust/nash", f"{log_name}.png") 
      elif oppType == "Uniform":
        file_path = os.path.join(f"./results/{log_dir}/robust/uniform", f"{log_name}.png") 
      else:
        print("これ出てる？")
        file_path = os.path.join(f"./results/{log_dir}/rew_plot", f"{log_name}.png") 
    else:
      os.makedirs(f"./make_results/{log_dir}", exist_ok=True)
      os.makedirs(f"./make_results/{log_dir}/rew_plot", exist_ok=True)
      file_path = os.path.join(f"./make_results/{log_dir}/rew_plot", f"{log_name}.png")

    plt.savefig(file_path)

  if moving_average :
    new_rews1_timing1 = []
    new_rews2_timing1 = []
    new_rews1_timing2 = []
    new_rews2_timing2 = []

    for i in range(len(rews1_timing2)-move_ave):
      new_rews1_timing1.append(sum(rews1_timing1[i:i+move_ave])/move_ave)
      new_rews2_timing1.append(sum(rews2_timing1[i:i+move_ave])/move_ave)
      new_rews1_timing2.append(sum(rews1_timing2[i:i+move_ave])/move_ave)
      new_rews2_timing2.append(sum(rews2_timing2[i:i+move_ave])/move_ave)

    rews1_timing1 = new_rews1_timing1
    rews2_timing1 = new_rews2_timing1
    rews1_timing2 = new_rews1_timing2
    rews2_timing2 = new_rews2_timing2
  
  if oppType == "Nash":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_Nash_{result_name}_エージェントA")
  elif oppType == "Uniform":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_Uni_{result_name}_エージェントA")
  else:
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"{result_name}_timing2")


