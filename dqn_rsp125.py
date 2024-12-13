from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from rsp125 import RSP125
from rsp125 import UniformAgent
from data_display import plot_rews,display_percentage_of_hand

class NashAgent(UniformAgent):
    def get_action(self, obs):
        return self.rng.choice((0, 1, 2), p=(2 / 17, 10 / 17, 5 / 17))

# 学習中のアクション履歴を記録するカスタムコールバック
class FullActionHistoryCallback(BaseCallback):
  def __init__(self):
    super(FullActionHistoryCallback, self).__init__()
    self.action_history = []
    self.rewards = []
    self.current_reward = 0

  def _on_step(self) -> bool:
    self.current_reward += self.locals['rewards'][0]
    if self.locals['dones']:
      self.rewards.append(self.current_reward)
      self.current_reward = 0

    if self.locals.get('actions') is not None:
      action = self.locals['actions']
      obs = self.locals['new_obs']
      if hasattr(self.training_env.envs[0], 'opp'):
        opp_action = self.training_env.envs[0].opp.get_action(obs)
        self.action_history.append((action, opp_action))
    return True

# クラスから手を出す戦略を作成
def create_new_agent(mod_action, class_name):
  return type(
    class_name,  # クラス名
    (UniformAgent,),  # 継承するベースクラス
    {
      # get_actionメソッドを上書き
      "get_action": lambda self, obs: mod_action.predict(obs, deterministic=True)[0],
      # "reset": lambda self: setattr(self, "rng", rng)  # rngを更新
    },
  )

learn_rate = 0.00005
num_iterations = 2000
def two_player():
  env = RSP125(goal=100, opp=NashAgent())
  action_callback_a = FullActionHistoryCallback()
  action_callback_b = FullActionHistoryCallback()
  playerA = DQN('MlpPolicy', env, learning_rate=learn_rate, verbose=0) # verbose=0で途中ログの出力制御
  playerB = DQN('MlpPolicy', env, learning_rate=learn_rate, verbose=0)
  # playerA = DQN('MlpPolicy', env, learning_rate=learn_rate, verbose=0, gradient_steps=100, train_freq=100) # verbose=0で途中ログの出力制御
  # playerB = DQN('MlpPolicy', env, learning_rate=learn_rate, verbose=0, gradient_steps=100, train_freq=100)
  
  # 学習を開始
  playerA.learn(total_timesteps=100, reset_num_timesteps=False, callback=action_callback_a)
  playerB.learn(total_timesteps=100, reset_num_timesteps=False, callback=action_callback_b)
  
  # 500回学習を繰り返す
  for i in range(num_iterations):
    if i % (num_iterations/10) == 0:
      print("#")
    aAgent = create_new_agent(playerA, "PlayerA")  # rngを渡す
    bAgent = create_new_agent(playerB, "PlayerB")  # rngを渡す
    
    playerA.set_env(RSP125(opp=bAgent(), goal=100))
    playerB.set_env(RSP125(opp=aAgent(), goal=100))
    
    # 各エージェントを1ステップ学習
    playerA.learn(total_timesteps=100, reset_num_timesteps=False, callback=action_callback_a)
    playerB.learn(total_timesteps=100, reset_num_timesteps=False, callback=action_callback_b)
  
  # モデルを保存
  playerA.save("playerA")
  playerB.save("playerB")
  
  # 行動履歴を取得
  action_history_a = action_callback_a.action_history
  action_history_b = action_callback_b.action_history

  # 平均報酬履歴
  reward_history_a = action_callback_a.rewards
  reward_history_b = action_callback_b.rewards
  
  return playerA, playerB, action_history_a, action_history_b, reward_history_a, reward_history_b


if __name__ == '__main__':
  
  # ２人のプレイヤー
  pl_a, pl_b, act_his_a, act_his_b, rew_his_a, rew_his_b = two_player()

  print("\n")
  id = np.random.RandomState(1000).randint(1, 10000)

  output_file_name = 'alternate_' + str(learn_rate) + "_" + str(num_iterations) + "_1.csv"
  display_percentage_of_hand(act_his_a, act_his_b, output_file_name)
  print(id)
  plot_rews(rew_his_a, rew_his_b, id, learn_rate)
