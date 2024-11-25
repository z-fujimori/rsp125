import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

def plot_rews(rews1,rews2):
  # x軸の値
  # x = np.arange(0, 1201)
  x = np.arange(1, min(len(rews1), len(rews2)) + 1)  # xは最小の長さに合わせる
  rews1 = rews1[:len(x)]
  rews2 = rews2[:len(x)]
  sum_rews = [(r1 + r2) / 2 for r1, r2 in zip(rews1, rews2)]

  # プロット
  plt.figure(figsize=(12, 6))  # グラフのサイズを指定
  plt.plot(x, sum_rews, label="Total rewerd", color="red")
  plt.plot(x, rews1, label="PlayerA", color="blue")
  plt.plot(x, rews2, label="PlayerB", color="orange")

  # グラフの設定
  plt.title("Line Graph of Two Datasets")
  plt.xlabel("Episode")
  plt.ylabel("Value")
  plt.legend()  # 凡例を追加
  plt.grid(True)  # グリッドを表示
  plt.tight_layout()  # レイアウトを調整

  plt.legend(framealpha=0.7) # 透過度
  # 表示
  plt.show()

def display_percentage_of_hand(hist_a, hist_b):
  percentage = np.zeros((3, 3), dtype=int) 
  total_pers = []

  for i, (hand_a, hand_b) in enumerate(zip(hist_a, hist_b)):
    hand_a = int(hand_a[0])
    hand_b = int(hand_b[0])
    percentage[hand_a][hand_b] += 1
    if i%100 == 99:
      total_pers.append(percentage)
      percentage = np.zeros((3, 3), dtype=int) 

  for i, per in enumerate(total_pers):
    print(f'{i}[A,B] [G,G]{per[0][0]} [G,C]{per[0][1]} [G,P]{per[0][2]} [C,G]{per[1][0]} [C,C]{per[1][1]} [C,P]{per[1][2]} [P,G]{per[2][0]} [P,C]{per[2][1]} [P,P]{per[2][2]}')
  
  
