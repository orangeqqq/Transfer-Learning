# Transfer-Learning
迁移学习的代码，
包含：

- Mingsheng Long的《Learning Transferable Features with Deep Adaptation Networks》提出的DAN模型。使用了PACS_Dataset:https://pan.baidu.com/s/1b6i9TlIQe0OAOirs5coVJA  提取码：9qw1

其中Tf2_DAN.py是一个独立的模块，可直接进行训练，其中没有进行复用已有的模型的参数。
其余文件为一组，在tl_DAN.py中复用了Cnn.py中所训练的模型参数。
