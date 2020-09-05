# 实验记录

## 20200823

版本回退道 20200623 517fb234 重新浮现当年测试记录

![result](train/20200623 517fb234/2020-06-23 07:43:42.png)

### 测试 train_20200823_base_on_20200623_517fb234
实验方法,版本回退道 **20200623 517fb234** 重新浮现当年测试记录,完全执行当时代码.进行运算\
包含实验:\
train_20200823_base_on_20200623_517fb234\
train_20200823_base_on_20200623_517fb234_2\
train_20200823_base_on_20200623_517fb234_3\
实验结果.分别为(其中较好的结果):

![result](train/train_20200823_base_on_20200623_517fb234/2020-08-24 00:02:39.png)
![result](train/train_20200823_base_on_20200623_517fb234_2/2020-08-24 16:27:01.png)
![result](train/train_20200823_base_on_20200623_517fb234_3/2020-08-24 16:30:51.png)

测试结果显示训练效果不太稳定,并不是每次都会又好的训练结果

## 20200824
### 测试 train_20200824_fac38f3d_reproduce_20200623_517fb234
实验方法:\
在 **20200623 517fb234** 的基础上,将最新版本的  env.py market.py 进行同步.测试是否依然可以产生与 **20200623 517fb234** 相同的训练效果.

实验包括及实验结果.分别为(其中较好的结果):
train_20200824_fac38f3d_reproduce_20200623_517fb234_1
![result](train/train_20200824_fac38f3d_reproduce_20200623_517fb234_1/2020-08-26 07:35:15.png)
![result](train/train_20200824_fac38f3d_reproduce_20200623_517fb234_1/2020-08-26 08:33:03.png)
train_20200824_fac38f3d_reproduce_20200623_517fb234_2
![result](train/train_20200824_fac38f3d_reproduce_20200623_517fb234_2/2020-08-26 09:52:23.png)
train_20200824_fac38f3d_reproduce_20200623_517fb234_3
![result](train/train_20200824_fac38f3d_reproduce_20200623_517fb234_3/2020-08-24 20:35:49.png)
![result](train/train_20200824_fac38f3d_reproduce_20200623_517fb234_3/2020-08-26 10:10:14.png)
train_20200824_fac38f3d_reproduce_20200623_517fb234_4
![result](train/train_20200824_fac38f3d_reproduce_20200623_517fb234_4/2020-08-25 19:53:00.png)

## 20200825
### 测试 train_20200825_1aa8dfec_reproduce_20200623_517fb234
实验方法:\
将 train_drl.py, common.uitls.py, common.metrics.py 等文件同步道最新版本后,继续浮现 **20200623_517fb234**

实验包括及实验结果.分别为(其中较好的结果):
train_20200825_1aa8dfec_reproduce_20200623_517fb234_1
![result](train/train_20200825_1aa8dfec_reproduce_20200623_517fb234_1/train_20200825_1aa8dfec_reproduce_20200623_517fb234_1/images/0351_2020-08-26 15:51:47_episode_final_plot.png)
train_20200825_1aa8dfec_reproduce_20200623_517fb234_2
![result](train/train_20200825_84224e73_reproduce_20200623_517fb234_2/train_20200825_1aa8dfec_reproduce_20200623_517fb234_2/images/0053_2020-08-27 00:38:14_episode_final_plot.png)
train_20200825_1aa8dfec_reproduce_20200623_517fb234_3
![result](train/train_20200825_84224e73_reproduce_20200623_517fb234_3/train_20200825_1aa8dfec_reproduce_20200623_517fb234_3/images/0133_2020-08-27 06:07:14_episode_final_plot.png)
train_20200825_1aa8dfec_reproduce_20200623_517fb234_4
![result](train/train_20200825_84224e73_reproduce_20200623_517fb234_4/train_20200825_1aa8dfec_reproduce_20200623_517fb234_4/images/0041_2020-08-26 23:51:29_episode_final_plot.png)
![result](train/train_20200825_84224e73_reproduce_20200623_517fb234_4/train_20200825_1aa8dfec_reproduce_20200623_517fb234_4/images/0038_2020-08-26 23:38:21_episode_final_plot.png)

## 实验结论
20200823,20200824,20200825三组训练均训练300轮
1) 实际上出现最好结果的轮数往往在前一半,后一半均出现了过度训练以至于失效的状态
2) 训练结果显示 env.py market.py 以及 train_drl.py, common.uitls.py, common.metrics.py 文件同步更新后还是可以起到同样的训练效果的
因此,训练结果的好坏更多的取决于网络的设计情况以及多次训练出现的随机结果.需要累计一定数量的训练才有一定的概率达到训练预期结果.
3) 因此如何判断模型有效,以及如何更高概率的产生有效模型成为下一步训练的重点
4) 模型训练期间参数无法保存的问题,需要后续寻找方法解决.
