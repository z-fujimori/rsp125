import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import csv

def plot_rews(rews1,rews2,id,learning_rate):
  # x軸の値
  # x = np.arange(0, 1201)
  x = np.arange(1, min(len(rews1), len(rews2)) + 1)  # xは最小の長さに合わせる
  rews1 = rews1[:len(x)]
  rews2 = rews2[:len(x)]
  sum_rews = [(r1 + r2) / 2 for r1, r2 in zip(rews1, rews2)]

  # プロット
  plt.figure(figsize=(12, 6))  # グラフのサイズを指定
  plt.plot(x, sum_rews, label="Total rewerd", color="gray")
  plt.plot(x, rews1, label="PlayerA", color="blue")
  plt.plot(x, rews2, label="PlayerB", color="orange")

  # グラフの設定
  plt.title(f'id:{id}  learn_rate:{learning_rate}')
  plt.xlabel("Episode")
  plt.ylabel("Value")
  plt.legend()  # 凡例を追加
  plt.grid(True)  # グリッドを表示
  plt.tight_layout()  # レイアウトを調整

  plt.legend(framealpha=0.7) # 透過度
  # 表示
  plt.show()

def display_percentage_of_hand(hist_a, hist_b, output_file_name='output.csv'):
  percentage = np.zeros((3, 3), dtype=int) 
  total_pers = []

  for i, (hand_a, hand_b) in enumerate(zip(hist_a, hist_b)):
    hand_a = int(hand_a[0])
    hand_b = int(hand_b[0])
    percentage[hand_a][hand_b] += 1
    if i%100 == 99:
      total_pers.append(percentage)
      percentage = np.zeros((3, 3), dtype=int) 
  
  # ターミナルに表示
  # for i, per in enumerate(total_pers):
  #   print(f'{i+1}')
  #   print(f'a\\b G | C | P |')
  #   print(f'G | {per[0][0]:02}| {per[0][1]:02}| {per[0][2]:02}|')
  #   print(f'C | {per[1][0]:02}| {per[1][1]:02}| {per[1][2]:02}|')
  #   print(f'P | {per[2][0]:02}| {per[2][1]:02}| {per[2][2]:02}|')

  # CSV保存部分
  with open(output_file_name, mode='w', newline='') as file:
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
