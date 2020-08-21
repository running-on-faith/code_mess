# 实验

## 思路
此前做法:

> LSTM 整理 factor 因子 -> 将 rr flag 表示为repeat 扩展后 concatenate 成为 3 * 93 的矩阵-> conv卷积操作 -> dualing -> 输出

现在考虑回归见到逻辑,并首先在 mock数据上实现可训练优化

> LSTM 整理 factor 因子 ->
> 直接与 rr factor 拼接 ->
> 全连接 -> 输出