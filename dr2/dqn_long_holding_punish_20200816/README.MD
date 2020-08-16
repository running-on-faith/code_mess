# 实验

## 改进思路
相较于 dqn_long_holding_punish, 将 holding_period 以 repeat 方式 combine 到一个矩阵里.

新的思路希望 holding_period 成为一个独立的 input,等 combine 后的矩阵处理完,变成1维数据后在进行二次 combine