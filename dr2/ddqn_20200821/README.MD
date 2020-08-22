# 实验

## 思路
此前做法:

> LSTM 整理 factor 因子 -> 将 rr flag 表示为repeat 扩展后 concatenate 成为 3 * 93 的矩阵-> conv卷积操作 -> dualing -> 输出

现在考虑回归见到逻辑,并首先在 mock数据上实现可训练优化

> LSTM 整理 factor 因子 ->
> 直接与 rr factor 拼接 ->
> 全连接 -> 输出

---
###  一. 测试: train_mock_20200821_3b9cd7fd
目的: 是在模拟行情中确认网络的有效性.

如果网络在sin模拟行情中展开有效的训练,达成净值不断上涨的情况,则说明改网络至少可以处理简单行情的情况

测试结果: 无效
#### 1. 20200821_lstm_no_punish_mock_96e9809f
![result](train_mock_20200821_3b9cd7fd/20200821_lstm_no_punish_mock_96e9809f/images/0054_2020-08-21 15:07:55_episode_final_plot.png)

#### 2. 20200821_lstm_no_punish_multiplier10_gamma0_mock_79c42345
![result](train_mock_20200821_3b9cd7fd/20200821_lstm_no_punish_multiplier10_gamma0_mock_79c42345/images/0033_2020-08-22 00:55:31_episode_final_plot.png)

#### 3. 20200822_lstm_no_punish_multiplier10_gamma10_mock_79c42345
![result](train_mock_20200821_3b9cd7fd/20200822_lstm_no_punish_multiplier10_gamma10_mock_79c42345/images/0022_2020-08-22 12:31:16_episode_final_plot.png)

#### 4. 20200822_lstm_fc2_no_punish_multiplier10_gamma10_mock_e52009a4

等待结果


#### final. 测试结果显示,目前尚未找到有效的网络

---
### 二. 测试 reproduce_20200623_517fb234_at_20200822_356111be
思路: dr2/dqn20200209/train/20200623_517fb234 显示 dqn训练存在有效性
![result](../dqn20200209/train/20200623 517fb234/2020-06-23 07:43:42.png)

测试 dr2/ddqn_20200821/reproduce_20200623_517fb234_at_20200822_07096044 目的是在现有的版本上重现该训练的成果

#### 1. 测试 reproduce_20200623_517fb234_at_20200822_07096044
关键网络参数:
> "fc_layer_params": [100],\
  'batch_normalization': False,\
  "activation_fn": "tanh" \

关键训练参数:
>loss_fn = _huber_loss\
gamma = 1.0,\
train_count_per_loop=10,

#### 2. 测试 train_20200822_39d511da_mock
与 reproduce_20200623_517fb234_at_20200822_07096044 参数一致
只是使用模拟行情


#### 测试结果:

---


