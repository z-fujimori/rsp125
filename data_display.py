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
from PIL import Image
import io

# フォントを日本語対応のものに設定
rcParams['font.family'] = 'Osaka'
plt.rcParams['font.family'] = 'Osaka'

move_ave = 100 # 移動平均

def plot_rews(rews1_timing1, rews2_timing1, rews1_timing2, rews2_timing2, result_name='output', run_time_log='--', num_trials=10000, step=1,is_save_mode=True, moving_average=True, move_ave_num=100, oppType=None):
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
    elif oppType == "TendR":
      np.save(f"./results/{result_name}/robust/tendR/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/tendR/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/tendR/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/tendR/rewsUni_mod1",rews2_timing2)
    elif oppType == "TendC":
      np.save(f"./results/{result_name}/robust/tendC/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/tendC/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/tendC/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/tendC/rewsUni_mod1",rews2_timing2)
    elif oppType == "TendP":
      np.save(f"./results/{result_name}/robust/tendP/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/tendP/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/tendP/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/tendP/rewsUni_mod1",rews2_timing2)
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
    elif oppType == "TendR":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="TendR", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "TendC":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="TendC", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "TendP":
      plt.plot(x, rews1, label="DQN", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="TendP", color="orange", alpha=0.6, linewidth=12)
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
      match oppType:
        case "Nash": file_path = os.path.join(f"./results/{log_dir}/robust/nash", f"{log_name}.png") 
        case "Uniform": file_path = os.path.join(f"./results/{log_dir}/robust/uniform", f"{log_name}.png") 
        case "TendR": file_path = os.path.join(f"./results/{log_dir}/robust/tendR", f"{log_name}.png") 
        case "TendC": file_path = os.path.join(f"./results/{log_dir}/robust/tendC", f"{log_name}.png") 
        case "TendP": file_path = os.path.join(f"./results/{log_dir}/robust/tendP", f"{log_name}.png") 
        case "R": file_path = os.path.join(f"./results/{log_dir}/robust/R", f"{log_name}.png") 
        case "C": file_path = os.path.join(f"./results/{log_dir}/robust/C", f"{log_name}.png") 
        case "P": file_path = os.path.join(f"./results/{log_dir}/robust/P", f"{log_name}.png") 
        case _: file_path = os.path.join(f"./results/{log_dir}/rew_plot", f"{log_name}.png") 
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

  if oppType == None:
    save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"{result_name}_timing1")
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"{result_name}_timing2")
  else:
    save_plot_rews(rews1_timing1, rews2_timing1, result_name, f"rew_{result_name}_エージェントA")
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_{result_name}_エージェントB")


def two_lines_rew_plot(rew1_data1, rew1_data2, rew1_name1, rew1_name2, rew2_data1, rew2_data2, rew2_name1, rew2_name2, result_name="fig", num_trials=10000, move_ave_num=100):
  mpl.rcParams['font.family'] = 'Osaka'
  # データが短すぎる場合の対策
  if len(rew1_data1) <= move_ave_num or len(rew2_data1) <= move_ave_num:
    print("データが短すぎます。move_ave_numを調整してください。")
    return
  
  # 移動平均の計算
  new_rew1_data1 = [sum(rew1_data1[i:i+move_ave_num])/move_ave_num for i in range(len(rew1_data1)-move_ave_num)]
  new_rew1_data2 = [sum(rew1_data2[i:i+move_ave_num])/move_ave_num for i in range(len(rew1_data2)-move_ave_num)]
  new_rew2_data1 = [sum(rew2_data1[i:i+move_ave_num])/move_ave_num for i in range(len(rew2_data1)-move_ave_num)]
  new_rew2_data2 = [sum(rew2_data2[i:i+move_ave_num])/move_ave_num for i in range(len(rew2_data2)-move_ave_num)]
  
  rew1_data1, rew1_data2 = new_rew1_data1, new_rew1_data2
  rew2_data1, rew2_data2 = new_rew2_data1, new_rew2_data2

  # 長さの確認
  if not (len(rew1_data1) == len(rew1_data2) == len(rew2_data1) == len(rew2_data2)):
    print("データの長さが一致していません")
    return

  # グラフ作成
  x = np.arange(1, len(rew1_data1) + 1)
  mpl.rcParams.update({'font.size': 40})
  # plt.tick_params(axis='both', labelsize=4)

  fig, axes = plt.subplots(2, 1, figsize=(20, 15), sharex=True)

  # 軸ラベルフォーマッター
  def multiply_by_five(x, pos):
    return f'{x * 1:.0f}'

  # グラフ1
  axes[0].plot(x, rew1_data1, label=f"{rew1_name1}", color="blue", alpha=0.7, linewidth=7)
  axes[0].plot(x, rew1_data2, label=f"{rew1_name2}", color="orange", alpha=0.7, linewidth=7)
  axes[0].set_ylabel('合計得点')
  axes[0].axhline(y=250, color='r', linestyle='--', linewidth=5)
  axes[0].legend(framealpha=0.7, fontsize=35)
  axes[0].text(-0.1, 0.9, 'A', transform=axes[0].transAxes, fontsize=50, fontweight='bold')

  # グラフ2
  axes[1].plot(x, rew2_data1, label=f"{rew2_name1}", color="blue", alpha=0.7, linewidth=7)
  axes[1].plot(x, rew2_data2, label=f"{rew2_name2}", color="orange", alpha=0.7, linewidth=7)
  axes[1].set_xlabel('trial')
  axes[1].set_ylabel('合計得点')
  axes[1].axhline(y=250, color='r', linestyle='--', linewidth=5)
  axes[1].legend(framealpha=0.7, fontsize=35)
  axes[1].text(-0.1, 0.9, 'B', transform=axes[1].transAxes, fontsize=50, fontweight='bold')

  # 共通設定
  for ax in axes:
    ax.xaxis.set_major_formatter(FuncFormatter(multiply_by_five))
    ax.set_xlim(0, len(rew1_data1))
    ax.set_ylim(1, 340)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    ax.minorticks_on()
    ax.grid(which='minor', linestyle='-', linewidth=1, alpha=0.4)
    ax.grid(which='major', linestyle='-', linewidth=3, alpha=0.7)

  plt.xticks(rotation=90)

  # 凡例の線の太さを変更
  # for ax in axes:
  #   leg = ax.legend()
  #   for line in leg.get_lines():
  #     line.set_linewidth(15)
    # for font in leg.get_fonts():
    #   font.set_fontsize(5)

  # 余白を調整
  plt.subplots_adjust(hspace=0.05, top=0.95, bottom=0.15)

  # 1. グラフを作成し、メモリ上に画像を保存
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png', dpi=50)
  # 2. メモリ上の画像データをPDFに変換
  buffer.seek(0)  # バッファの先頭に戻す
  image = Image.open(buffer)
  pdf_buffer = io.BytesIO()
  image.save(pdf_buffer, format="PDF", resolution=300)
  # 3. PDFデータをファイルに保存
  with open(f"./figs/{result_name}.pdf", "wb") as f:
    f.write(pdf_buffer.getvalue())
  # メモリのバッファをクローズ
  buffer.close()
  pdf_buffer.close()


def display_percentage_of_hand(hist_a_timing1, hist_b_timing1, hist_a_timing2, hist_b_timing2, result_name='output', run_time_log='--', moving_averae=True, oppType=None):

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
    match oppType:
      case "Nash": result_name = f"./results/{log_dir}/robust/nash/{log_name}.csv"
      case "Uniform": result_name = f"./results/{log_dir}/robust/uniform/{log_name}.csv"
      case "TendR": result_name = f"./results/{log_dir}/robust/tendR/{log_name}.csv"
      case "TendC": result_name = f"./results/{log_dir}/robust/tendC/{log_name}.csv"
      case "TendP": result_name = f"./results/{log_dir}/robust/tendP/{log_name}.csv"
      case "R": result_name = f"./results/{log_dir}/robust/R/{log_name}.csv"
      case "C": result_name = f"./results/{log_dir}/robust/C/{log_name}.csv"
      case "P": result_name = f"./results/{log_dir}/robust/P/{log_name}.csv"
      case _: result_name = f"./results/{log_dir}/hand_csv/{log_name}.csv"

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
    match oppType:
      case "Nash": file_path = os.path.join(f"./results/{log_dir}/robust/nash", f"{log_name}.png")
      case "Uniform": file_path = os.path.join(f"./results/{log_dir}/robust/uniform", f"{log_name}.png")
      case "TendR": file_path = os.path.join(f"./results/{log_dir}/robust/tendR", f"{log_name}.png")
      case "TendC": file_path = os.path.join(f"./results/{log_dir}/robust/tendC", f"{log_name}.png")
      case "TendP": file_path = os.path.join(f"./results/{log_dir}/robust/tendP", f"{log_name}.png")
      case "R": file_path = os.path.join(f"./results/{log_dir}/robust/R", f"{log_name}.png")
      case "C": file_path = os.path.join(f"./results/{log_dir}/robust/C", f"{log_name}.png")
      case "P": file_path = os.path.join(f"./results/{log_dir}/robust/P", f"{log_name}.png")
      case _: file_path = os.path.join(f"./results/{log_dir}/hand_csv", f"{log_name}.png")
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
  plt.figure(figsize=(65, 30))  # グラフのサイズを指定
  plt.plot(range(len(rew_g_g)), rew_g_g, label="A: g, B: g", color="brown", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_g_c, label="A: g, B: c", color="orange", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_g_p, label="A: g, B: p", color="blue", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_c_g, label="A: c, B: g", color="green", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_c_c, label="A: c, B: c", color="gray", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_c_p, label="A: c, B: p", color="pink", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_p_g, label="A: p, B: g", color="red", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_p_c, label="A: p, B: c", color="olive", alpha=0.6, linewidth=12)
  plt.plot(range(len(rew_g_g)), rew_p_p, label="A: p, B: p", color="purple", alpha=0.6, linewidth=12)

  # グラフの設定
  plt.title(f'{file_name}', fontsize=32)
  plt.xlabel("trial")
  plt.ylabel("count")
  plt.xticks(rotation=90) # 横軸のメモリの表記を縦書きにする

  # plt.legend(framealpha=0.7)  # 凡例を追加
  # x=250に太線を引く
  # plt.axvline(y=250, color='red', linewidth=12, linestyle='--')
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
  move_ave = 100
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
    elif oppType == "TendR":
      np.save(f"./results/{result_name}/robust/tendR/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/tendR/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/tendR/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/tendR/rewsUni_mod1",rews2_timing2)
    elif oppType == "TendC":
      np.save(f"./results/{result_name}/robust/tendC/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/tendC/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/tendC/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/tendC/rewsUni_mod1",rews2_timing2)
    elif oppType == "TendP":
      np.save(f"./results/{result_name}/robust/tendP/rews0_mod0",rews1_timing1)
      np.save(f"./results/{result_name}/robust/tendP/rewsUni_mod0",rews2_timing1)
      np.save(f"./results/{result_name}/robust/tendP/rews1_mod1",rews1_timing2)
      np.save(f"./results/{result_name}/robust/tendP/rewsUni_mod1",rews2_timing2)
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
    sum_rews = (np.array(rews1) + np.array(rews2)) / 2

    mpl.rcParams.update({'font.size': 128})

    # プロット
    plt.figure(figsize=(65, 30))  # グラフのサイズを指定
    # plt.plot(x, sum_rews, label="平均点", color="black", alpha=0.9, linewidth=12)
    if oppType == "Nash":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.5, linewidth=12)
      plt.plot(x, rews2, label="Nash", color="orange", alpha=0.5, linewidth=12)
    elif oppType == "Uniform":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.5, linewidth=12)
      plt.plot(x, rews2, label="uniform(1/3)", color="orange", alpha=0.5, linewidth=12)
    elif oppType == "TendR":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="TendR", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "TendC":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="TendC", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "TendP":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="TendP", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "R":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="R", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "C":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="C", color="orange", alpha=0.6, linewidth=12)
    elif oppType == "P":
      plt.plot(x, rews1, label="エージェント", color="blue", alpha=0.6, linewidth=12)
      plt.plot(x, rews2, label="P", color="orange", alpha=0.6, linewidth=12)
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
    plt.ylim(1,340)
    plt.xticks(rotation=90) # 横軸のメモリの表記を縦書きにする
    # plt.title(f'{log_name}_{run_time_log}min', fontsize=32)
    plt.xlabel("trial")
    plt.ylabel("合計得点")
    plt.legend(framealpha=0.7)  # 凡例を追加
    # y=250 の水平線を引く
    plt.axhline(y=250, color='r', linestyle='--', linewidth=8)
    # メジャー目盛り
    plt.gca().xaxis.set_major_locator(MultipleLocator(100))
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
    # leg.get_lines()[2].set_linewidth(20)
    # 保存
    if (is_save_mode):
      match oppType:
        case "Nash": file_path = os.path.join(f"./results/{log_dir}/robust/nash", f"{log_name}.png") 
        case "Uniform": file_path = os.path.join(f"./results/{log_dir}/robust/uniform", f"{log_name}.png") 
        case "TendR": file_path = os.path.join(f"./results/{log_dir}/robust/tendR", f"{log_name}.png") 
        case "TendC": file_path = os.path.join(f"./results/{log_dir}/robust/tendC", f"{log_name}.png") 
        case "TendP": file_path = os.path.join(f"./results/{log_dir}/robust/tendP", f"{log_name}.png") 
        case "R": file_path = os.path.join(f"./results/{log_dir}/robust/R", f"{log_name}.png") 
        case "C": file_path = os.path.join(f"./results/{log_dir}/robust/C", f"{log_name}.png") 
        case "P": file_path = os.path.join(f"./results/{log_dir}/robust/P", f"{log_name}.png") 
        case _: file_path = os.path.join(f"./results/{log_dir}/rew_plot", f"{log_name}.png") 
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
  elif oppType == "TendR":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_TendR_{result_name}_エージェントA")
  elif oppType == "TendC":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_TendC_{result_name}_エージェントA")
  elif oppType == "TendP":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_TendP_{result_name}_エージェントA")
  elif oppType == "R":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_R_{result_name}_エージェントA")
  elif oppType == "C":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_C_{result_name}_エージェントA")
  elif oppType == "P":
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"rew_P_{result_name}_エージェントA")
  else:
    save_plot_rews(rews1_timing2, rews2_timing2, result_name, f"{result_name}_timing2")

def robust_evaluation(result_name, dqn_nash, opp_nash, dqn_r, opp_r, dqn_c, opp_c, dqn_p, opp_p):
  # 末尾500回の平均を計算
  def ave_of_last(array, last_num=500):
    return np.mean(array[-last_num:])
  ave_dqn_nash = ave_of_last(dqn_nash)
  ave_opp_nash = ave_of_last(opp_nash)
  ave_dqn_r = ave_of_last(dqn_r)
  ave_opp_r = ave_of_last(opp_r)
  ave_dqn_c = ave_of_last(dqn_c)
  ave_opp_c = ave_of_last(opp_c)
  ave_dqn_p = ave_of_last(dqn_p)
  ave_opp_p = ave_of_last(opp_p)

  data = {
    'Nash': [ave_dqn_nash, ave_opp_nash],
    'OnlyR': [ave_dqn_r, ave_opp_r],
    'OnlyS': [ave_dqn_c, ave_opp_c],
    'OnlyP': [ave_dqn_p, ave_opp_p]
  }

  labels = list(data.keys())
  values = np.array(list(data.values()))
  # 棒グラフの幅と位置を設定
  x = np.arange(len(labels))  # X軸の位置
  width = 0.35  # 棒の幅
  # 棒グラフを描画
  mpl.rcParams.update({'font.size': 25})
  fig, ax = plt.subplots()
  bar1 = ax.bar(x - width/2, values[:, 0], width, label='機械学習エージェント', color='blue')
  bar2 = ax.bar(x + width/2, values[:, 1], width, label='固定戦略エージェント', color='orange')
  
  ax.set_ylim(0, 390)
  ax.yaxis.set_major_locator(MultipleLocator(50))
  ax.yaxis.set_minor_locator(MultipleLocator(10))
  ax.grid(axis='y', which='minor', linestyle='-', linewidth=0.5, alpha=0.3)
  ax.grid(axis='y', which='major', linestyle='-', linewidth=1, alpha=0.6)
  # ax.set_xlabel('x軸ラベル')
  ax.set_ylabel('合計得点')
  # ax.set_title('タイトル')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend(framealpha=0.7, fontsize=14)

  # グラフを表示
  plt.tight_layout()
  # plt.show()

  # 1. グラフを作成し、メモリ上に画像を保存
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png', dpi=250)
  # 2. メモリ上の画像データをPDFに変換
  buffer.seek(0)  # バッファの先頭に戻す
  image = Image.open(buffer)
  pdf_buffer = io.BytesIO()
  image.save(pdf_buffer, format="PDF", resolution=300)
  # 3. PDFデータをファイルに保存
  with open(f"./figs/{result_name}.pdf", "wb") as f:
    f.write(pdf_buffer.getvalue())
  # メモリのバッファをクローズ
  buffer.close()
  pdf_buffer.close()

def two_col_robust_evaluation(result_name, dqn_nash_0, opp_nash_0, dqn_r_0, opp_r_0, dqn_c_0, opp_c_0, dqn_p_0, opp_p_0, dqn_nash_1, opp_nash_1, dqn_r_1, opp_r_1, dqn_c_1, opp_c_1, dqn_p_1, opp_p_1):
  def ave_of_last(array, last_num=500):
    return np.mean(array[-last_num:])
  ave_dqn_nash_0 = ave_of_last(dqn_nash_0)
  ave_opp_nash_0 = ave_of_last(opp_nash_0)
  ave_dqn_r_0 = ave_of_last(dqn_r_0)
  ave_opp_r_0 = ave_of_last(opp_r_0)
  ave_dqn_c_0 = ave_of_last(dqn_c_0)
  ave_opp_c_0 = ave_of_last(opp_c_0)
  ave_dqn_p_0 = ave_of_last(dqn_p_0)
  ave_opp_p_0 = ave_of_last(opp_p_0)
  ave_dqn_nash_1 = ave_of_last(dqn_nash_1)
  ave_opp_nash_1 = ave_of_last(opp_nash_1)
  ave_dqn_r_1 = ave_of_last(dqn_r_1)
  ave_opp_r_1 = ave_of_last(opp_r_1)
  ave_dqn_c_1 = ave_of_last(dqn_c_1)
  ave_opp_c_1 = ave_of_last(opp_c_1)
  ave_dqn_p_1 = ave_of_last(dqn_p_1)
  ave_opp_p_1 = ave_of_last(opp_p_1)

  datas = [
    {
      'Nash': [ave_dqn_nash_0, ave_opp_nash_0],
      'OnlyR': [ave_dqn_r_0, ave_opp_r_0],
      'OnlyS': [ave_dqn_c_0, ave_opp_c_0],
      'OnlyP': [ave_dqn_p_0, ave_opp_p_0]
    },
    {
      'Nash': [ave_dqn_nash_1, ave_opp_nash_1],
      'OnlyR': [ave_dqn_r_1, ave_opp_r_1],
      'OnlyS': [ave_dqn_c_1, ave_opp_c_1],
      'OnlyP': [ave_dqn_p_1, ave_opp_p_1]
    }
  ]
  graph_labels = ['A', 'B']

  mpl.rcParams.update({'font.size': 30})
  fig, axes = plt.subplots(1, 2, figsize=(15,8), sharey=True)

  for ax, data, label in zip(axes, datas, graph_labels):
    labels = list(data.keys())
    values = np.array(list(data.values()))
    # 棒グラフの幅と位置を設定
    x = np.arange(len(labels))  # X軸の位置
    width = 0.4  # 棒の幅
    # 棒グラフを描画
    ax.bar(x - width/2, values[:, 0], width, label='機械学習エージェント', color='blue')
    ax.bar(x + width/2, values[:, 1], width, label='固定戦略エージェント', color='orange')
    
    ax.set_ylim(0, 390)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis='y', which='minor', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(axis='y', which='major', linestyle='-', linewidth=1, alpha=0.6)
    # ax.set_xlabel('x軸ラベル')
    # ax.set_ylabel('合計得点')
    # ax.set_title('タイトル')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.7, fontsize=20)

    ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=30, fontweight='bold', ha='left') # 左上の文字の設定

  axes[0].set_ylabel('合計得点')
  # グラフを表示
  plt.tight_layout()
  # plt.show()

  # 1. グラフを作成し、メモリ上に画像を保存
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png', dpi=250)
  # 2. メモリ上の画像データをPDFに変換
  buffer.seek(0)  # バッファの先頭に戻す
  image = Image.open(buffer)
  pdf_buffer = io.BytesIO()
  image.save(pdf_buffer, format="PDF", resolution=300)
  # 3. PDFデータをファイルに保存
  with open(f"./figs/{result_name}.pdf", "wb") as f:
    f.write(pdf_buffer.getvalue())
  # メモリのバッファをクローズ
  buffer.close()
  pdf_buffer.close()

def three_col_robust_evaluation(result_name, dqn_nash, opp_nash, dqn_r, opp_r, dqn_c, opp_c, dqn_p, opp_p, high_dqn_nash, high_opp_nash, high_dqn_r, high_opp_r, high_dqn_c, high_opp_c, high_dqn_p, high_opp_p, low_dqn_nash, low_opp_nash, low_dqn_r, low_opp_r, low_dqn_c, low_opp_c, low_dqn_p, low_opp_p):
  # 末尾500回の平均を計算
  def ave_of_last(array, last_num=500):
    return np.mean(array[-last_num:])
  ave_dqn_nash = ave_of_last(dqn_nash)
  ave_opp_nash = ave_of_last(opp_nash)
  ave_dqn_r = ave_of_last(dqn_r)
  ave_opp_r = ave_of_last(opp_r)
  ave_dqn_c = ave_of_last(dqn_c)
  ave_opp_c = ave_of_last(opp_c)
  ave_dqn_p = ave_of_last(dqn_p)
  ave_opp_p = ave_of_last(opp_p)
  ave_high_dqn_nash = ave_of_last(high_dqn_nash)
  ave_high_opp_nash = ave_of_last(high_opp_nash)
  ave_high_dqn_r = ave_of_last(high_dqn_r)
  ave_high_opp_r = ave_of_last(high_opp_r)
  ave_high_dqn_c = ave_of_last(high_dqn_c)
  ave_high_opp_c = ave_of_last(high_opp_c)
  ave_high_dqn_p = ave_of_last(high_dqn_p)
  ave_high_opp_p = ave_of_last(high_opp_p)
  ave_low_dqn_nash = ave_of_last(low_dqn_nash)
  ave_low_opp_nash = ave_of_last(low_opp_nash)
  ave_low_dqn_r = ave_of_last(low_dqn_r)
  ave_low_opp_r = ave_of_last(low_opp_r)
  ave_low_dqn_c = ave_of_last(low_dqn_c)
  ave_low_opp_c = ave_of_last(low_opp_c)
  ave_low_dqn_p = ave_of_last(low_dqn_p)
  ave_low_opp_p = ave_of_last(low_opp_p)

  datas = [
    {
      'Nash': [ave_dqn_nash, ave_opp_nash],
      'OnlyR': [ave_dqn_r, ave_opp_r],
      'OnlyS': [ave_dqn_c, ave_opp_c],
      'OnlyP': [ave_dqn_p, ave_opp_p]
    },
    {
      'Nash': [ave_high_dqn_nash, ave_high_opp_nash],
      'OnlyR': [ave_high_dqn_r, ave_high_opp_r],
      'OnlyS': [ave_high_dqn_c, ave_high_opp_c],
      'OnlyP': [ave_high_dqn_p, ave_high_opp_p]
    },
    {
      'Nash': [ave_low_dqn_nash, ave_low_opp_nash],
      'OnlyR': [ave_low_dqn_r, ave_low_opp_r],
      'OnlyS': [ave_low_dqn_c, ave_low_opp_c],
      'OnlyP': [ave_low_dqn_p, ave_low_opp_p]
    },
  ]
  graph_labels = ['A', 'B', 'C']

  mpl.rcParams.update({'font.size': 35})
  fig, axes = plt.subplots(1, 3, figsize=(20,8), sharey=True)

  for ax, data, label in zip(axes, datas, graph_labels):
    labels = list(data.keys())
    values = np.array(list(data.values()))
    # 棒グラフの幅と位置を設定
    x = np.arange(len(labels))  # X軸の位置
    width = 0.4  # 棒の幅
    # 棒グラフを描画
    ax.bar(x - width/2, values[:, 0], width, label='機械学習エージェント', color='blue')
    ax.bar(x + width/2, values[:, 1], width, label='固定戦略エージェント', color='orange')
    
    ax.set_ylim(0, 390)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis='y', which='minor', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(axis='y', which='major', linestyle='-', linewidth=1, alpha=0.6)
    # ax.set_xlabel('x軸ラベル')
    # ax.set_ylabel('合計得点')
    # ax.set_title('タイトル')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    # ax.set_xticklabels(labels)
    # ax.legend(framealpha=0.7, fontsize=20, bbox_to_anchor=(0.6, 0.5))
    ax.legend(framealpha=0.7, fontsize=20)

    ax.text(-0.1, 1.0, label, transform=ax.transAxes, fontsize=30, fontweight='bold', ha='left') # 左上の文字の設定

  plt.subplots_adjust(hspace=0.05)
  axes[0].set_ylabel('合計得点')
  # グラフを表示
  plt.tight_layout()
  # plt.show()

  # 1. グラフを作成し、メモリ上に画像を保存
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png', dpi=250)
  # 2. メモリ上の画像データをPDFに変換
  buffer.seek(0)  # バッファの先頭に戻す
  image = Image.open(buffer)
  pdf_buffer = io.BytesIO()
  image.save(pdf_buffer, format="PDF", resolution=300)
  # 3. PDFデータをファイルに保存
  with open(f"./figs/{result_name}.pdf", "wb") as f:
    f.write(pdf_buffer.getvalue())
  # メモリのバッファをクローズ
  buffer.close()
  pdf_buffer.close()
