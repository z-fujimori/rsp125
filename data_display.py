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

def plot_rews(rews1_timing1,rews2_timing1,rews1_timing2,rews2_timing2,result_name='output',run_time_log='--',num_trials=10000,step=5,is_save_mode=True, moving_average=True):
  if (is_save_mode):
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
    plt.plot(x, sum_rews, label="平均点", color="black", alpha=0.9)
    plt.plot(x, rews1, label="PlayerA", color="blue", alpha=0.5)
    plt.plot(x, rews2, label="PlayerB", color="orange", alpha=0.5)
    # plt.plot(x, rews1, label="機械学習エージェント", color="blue", alpha=0.5)
    # plt.plot(x, rews2, label="しっぺ返しエージェント", color="orange", alpha=0.5)

    # 横軸ラベルを変更するためのFormatter
    def multiply_by_five(x, pos):
        return f'{x * step:.0f}'  # 倍率step倍にし、小数点なしのフォーマット

    # グラフの設定
    # 軸設定
    plt.gca().xaxis.set_major_formatter(FuncFormatter(multiply_by_five))
    plt.xlim(0, num_trials//step)
    # plt.ylim(0,440)
    plt.xticks(rotation=90) # 横軸のメモリの表記を縦書きにする
    plt.title(f'{log_name}_{run_time_log}min', fontsize=32)
    plt.xlabel("Episode")
    plt.ylabel("Value")
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
      file_path = os.path.join(f"./results/{log_dir}/rew_plot", f"{log_name}.png") 
    else:
      os.makedirs(f"./make_results/{log_dir}", exist_ok=True)
      os.makedirs(f"./make_results/{log_dir}/rew_plot", exist_ok=True)
      file_path = os.path.join(f"./make_results/{log_dir}/rew_plot", f"{log_name}.png")

    plt.savefig(file_path)
      # 表示
      # plt.show()
    # plt.close()

  if moving_average :
    new_rews1_timing1 = []
    new_rews2_timing1 = []
    new_rews1_timing2 = []
    new_rews2_timing2 = []

    for i in range(len(rews1_timing1)-10):
      new_rews1_timing1.append(sum(rews1_timing1[i:i+10])/10)
      new_rews2_timing1.append(sum(rews2_timing1[i:i+10])/10)
      new_rews1_timing2.append(sum(rews1_timing2[i:i+10])/10)
      new_rews2_timing2.append(sum(rews2_timing2[i:i+10])/10)

    rews1_timing1 = new_rews1_timing1
    rews2_timing1 = new_rews2_timing1
    rews1_timing2 = new_rews1_timing2
    rews2_timing2 = new_rews2_timing2

  save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"{result_name}_timing1")
  save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"{result_name}_timing2")


def display_percentage_of_hand(hist_a_timing1, hist_b_timing1, hist_a_timing2, hist_b_timing2, result_name='output',run_time_log='--'):

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

    hand_plt = create_plt_data(hand_plot_hist[0][0],hand_plot_hist[0][1],hand_plot_hist[0][2],hand_plot_hist[1][0],hand_plot_hist[1][1],hand_plot_hist[1][2],hand_plot_hist[2][0],hand_plot_hist[2][1],hand_plot_hist[2][2])
    file_path = os.path.join(f"./results/{log_dir}/hand_csv", f"{log_name}.png")
    hand_plt.savefig(file_path)

  save_hand_hist(hist_a_timing1, hist_b_timing1, result_name, f"{result_name}_timing1")
  save_hand_hist(hist_a_timing2, hist_b_timing2, result_name, f"{result_name}_timing2")


def plot_hand_hist_csv(csv_file):
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
  
  plt = create_plt_data(hand_plot_hist[0][0],hand_plot_hist[0][1],hand_plot_hist[0][2],hand_plot_hist[1][0],hand_plot_hist[1][1],hand_plot_hist[1][2],hand_plot_hist[2][0],hand_plot_hist[2][1],hand_plot_hist[2][2])

  plt.show()


def create_plt_data(rew_g_g, rew_g_c, rew_g_p, rew_c_g, rew_c_c, rew_c_p, rew_p_g, rew_p_c, rew_p_p, file_name="hist"):

  plt.figure(figsize=(100, 10))  # グラフのサイズを指定
  plt.plot(range(len(rew_g_g)), rew_g_g, label="A: g, B: g", color="brown", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_g_c, label="A: g, B: c", color="orange", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_g_p, label="A: g, B: p", color="blue", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_c_g, label="A: c, B: g", color="green", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_c_c, label="A: c, B: c", color="gray", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_c_p, label="A: c, B: p", color="pink", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_p_g, label="A: p, B: g", color="red", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_p_c, label="A: p, B: c", color="olive", alpha=0.6)
  plt.plot(range(len(rew_g_g)), rew_p_p, label="A: p, B: p", color="purple", alpha=0.6)

  # グラフの設定
  plt.title(f'{file_name}')
  plt.xlabel("time")
  plt.ylabel("count")
  # plt.legend(framealpha=0.7)  # 凡例を追加
  # x=250に太線を引く
  # plt.axvline(y=250, color='red', linewidth=2.5, linestyle='--')
  # グリッド目盛り設定
  plt.gca().xaxis.set_major_locator(MultipleLocator(25))
  plt.gca().yaxis.set_major_locator(MultipleLocator(5))
  plt.grid()  # グリッドを表示
  plt.tight_layout()  # レイアウトを調整

  plt.legend(framealpha=0.6) # 透過度
  # 表示
  return plt
