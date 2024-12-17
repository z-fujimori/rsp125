import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import csv
import os

def plot_rews(rews1_timing1,rews2_timing1,rews1_timing2,rews2_timing2,result_name,learning_rate):
  
  def save_plot_rews(rews1, rews2, log_dir, log_name):
    x = np.arange(1, min(len(rews1), len(rews2)) + 1)  # xは最小の長さに合わせる
    rews1 = rews1[:len(x)]
    rews2 = rews2[:len(x)]
    sum_rews = [(r1 + r2) / 2 for r1, r2 in zip(rews1, rews2)]

    # プロット
    plt.figure(figsize=(60, 30))  # グラフのサイズを指定
    plt.plot(x, sum_rews, label="Total rewerd", color="gray")
    plt.plot(x, rews1, label="PlayerA", color="blue")
    plt.plot(x, rews2, label="PlayerB", color="orange")

    # グラフの設定
    plt.title(f'{log_name}')
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend(framealpha=0.7)  # 凡例を追加
    plt.grid(True)  # グリッドを表示
    plt.tight_layout()  # レイアウトを調整

    plt.legend(framealpha=0.7) # 透過度
    # 表示
    plt.show()
    # 保存
    file_path = os.path.join(f"./results/{log_dir}/rew_plot", log_name)
    plt.savefig(file_path)
    plt.close()

  save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"{result_name}_timing1.png")
  save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"{result_name}_timing2.png")

def display_percentage_of_hand(hist_a_timing1, hist_b_timing1, hist_a_timing2, hist_b_timing2, result_name='output'):

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
    result_name = f"./results/{log_dir}/hand_csv/{log_name}"
    with open(result_name, mode='w', newline='') as file:
      writer = csv.writer(file)
      # ヘッダーを書き込む
      writer.writerow(["Round", "a\\b", "G", "C", "P"])
      for i, per in enumerate(total_pers):
        # ラウンド名
        writer.writerow([f'Round {i+1}'])
        # データ行を書き込む
        writer.writerow(["", "G", *per[0]])
        writer.writerow(["", "C", *per[1]])
        writer.writerow(["", "P", *per[2]])
        writer.writerow([])  # 空行を挿入して次のラウンドと区切る
  
  save_hand_hist(hist_a_timing1, hist_b_timing1, result_name, f"{result_name}_timing1.csv")
  save_hand_hist(hist_a_timing2, hist_b_timing2, result_name, f"{result_name}_timing2.csv")
