# 飞桨论文复现挑战赛（第四期）参赛心得

## 赛事介绍及参赛目的
### 赛事介绍
[飞桨论文复现挑战赛（第四期）](https://aistudio.baidu.com/aistudio/competition/detail/106)要求选手在指定时间内使用飞桨完成论文复现，考察复现模型的精度。

### 参赛原因
- 了解下深度学习
- 有百度算力支持
- 丰厚奖金

## 参赛记录
我所选择的论文是：[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/pdf/1511.00561.pdf)  

主要参考了以下文档：
- [PaddlePaddle API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)
- [PaddlePaddle 2.1 使用教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
- [PaddleSeg 2.2 使用文档](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/configs)
- 以及万能的 [百度](www.baidu.com)  

主要面临的问题及相应解决方案如下：
- PaddlePaddle 网上资料较少
  - 与百度工程师交流
- PaddlePaddle 代码有些bug
  - 定位bug并且暂时修改问题代码
- 算力需求较大
  - Ai Studio 提供 V100 算力
- 一个人好孤单～
  - 找队友！
## 参赛感想
在此我将结合个人经验对深度学习框架国产化作出期待。

深度学习是由三大力量同时驱动的：算力、数据和算法。深度学习是机器学习的一个分支，他的优点是能够学习深层特征，甚至能超过人类的表现，如 ResNet 在 ImageNet 上的表现超过了人类。  

目前为止，深度学习的舞台主要是学术界和工业界。学术界目前以 PyTorch 框架为主，工业界有 Caffe2 和 TensorFlow。国产深度学习框架目前并没有较大的影响力。

“科技是第一生产力”，在遭受了“新片断供”后，我们发现核心技术必须由自己掌握。自此，国家对高科技产业更为重视。

在上面提到过，算力、数据和算法是深度学习的三大依赖。中国巨大的人口基数决定了数据层面我们具有优势，算力层面，华为的昇腾系列芯片就达到了世界领先水平。前有“MatLab”断供哈工大的教训，我们意识到软件层面我们也必须占有主动权，于是负着众望，飞浆 PaddlePaddle 框架走入人们的视线。

PyTorch 流行于学界，很大程度归因于其较大的开发者社区。因此，PaddlePaddle 想要成功，需要拥有其用户群体，而此次的飞桨论文复现挑战赛首先是吸引大家关注 PaddlePaddle ，然后利用其优势吸引万千开发者。

在这次比赛中，我也体验了 PaddlePaddle 框架，它可以满足“快速开发”和“快速个性化开发”的需求，以后也会关注和使用 PaddlePaddle 。

在此，预祝 PaddlePaddle 飞浆框架越办越好！


## Further information
|  information   | description  |
|  ----  | ----  |
| Author  | chen jiejun |
| E-mail | stuartchen2018@outlook.com |
| Date  | 2021.8.27 |
