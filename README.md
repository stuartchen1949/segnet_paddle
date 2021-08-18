# SegNet
SegNet
[SegNet](#segnet)
   * [一、简介](#一简介)
   * [二、复现精度](#二复现精度)
   * [三、数据集](#三数据集)
   * [四、环境依赖](#四环境依赖)
   * [五、快速开始](#五快速开始)
      * [快速训练：](#快速训练)
      * [快速预测：](#快速预测)
   * [六、代码结构与详细说明](#六代码结构与详细说明)
   * [七、模型信息](#七模型信息)
## 一、简介

SegNet，用来实现语义分割的深度学习神经网络模型  
论文：<https://github.com/yassouali/pytorch-segmentation>  
已在 AI Studio 上进行开源，可以在线运行：<https://aistudio.baidu.com/aistudio/projectdetail/2276343>

## 二、复现精度
在 camvid 11类数据集上 miou = 0.601  
ps：由于使用paddleseg套件开发，void类也进行了计算，所以我们对12类（包含了 void 类，训练时为了获得更高精度，label 记为12类）的miou进行等效：  
等效miou = （原miou *12 - iou（ignore_label））/ 11，我们的12类miou等效后达到11类miou=0.601的标准  
当然，也可以在训练完成后将类别数改为11类进行预测计算 miou。

## 三、数据集
本次数据集使用的是 camvid，可以在 AI Studio 上下载

链接：<https://aistudio.baidu.com/aistudio/datasetdetail/79232>

这个版本来自于：<https://www.kaggle.com/naureenmohammad/camvid-dataset>,其中分辨率较低，为 480x360，其中标签的id范围是0到11

数据集大小：
- 训练集样本数量：367
- 验证集样本数量：101
- 测试集样本数量：233

格式如下：
- test           测试集图像（.png）
- testannot      测试集标签（.png）
- train          训练集图像（.png）
- trainannot     训练集标签（.png）
- val            验证集图像（.png）
- valannot       验证集标签（.png）

## 四、环境依赖
- 硬件
  - GPU
  - CPU
- 软件
  - PaddlePaddle = 2.1
  - PaddleSeg

## 五、快速开始
### 快速训练：
```
cd PaddleSeg
python train.py \
       --config config.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```
其中参数说明如下：
- config ：设置文件路径
- do_eval ：在训练过程中进行验证
- use_vdl ：使用 VisualDl
- save_interval ：模型保存周期
- save_dir ：模型及日志保存位置

### 快速预测：
```
python data/PaddleSeg/predict.py \
       --config config.yml \
       --model_path output_bs_8——pre/best_model/model.pdparams \
       --image_path data/PaddleSeg/camvid/test \
       --save_dir output/result
```
其中参数说明如下：
- config ：设置文件路径
- model_path ：使用的模型位置
- image_path ：需要测试的数据（可以是单个文件或文件夹）
- save_dir ：输出保存位置

## 六、代码结构与详细说明
本项目使用 PaddlePaddle 实现，代码结构与 PaddleSeg 类似，只需要个性化的在 `config.yml` 中进行设置

下面介绍 `config.yml` 的参数设置：
```
batch_size: 4  #设定batch_size的值即为迭代一次送入网络的图片数量，一般显卡显存越大，batch_size的值可以越大
iters: 1000    #模型迭代的次数

train_dataset: #训练数据设置
  type: Dataset #选择数据集格式
  dataset_root: data/PaddleSeg/camvid #选择数据集路径
  train_path: data/PaddleSeg/camvid/train_list.txt #选择数据集list
  num_classes: 12 #指定目标的类别个数（背景也算为一类）
  transforms: #数据预处理/增强的方式
    - type: Resize #送入网络之前需要进行resize
      target_size: [512, 512] #将原图resize成512*512在送入网络
    - type: RandomHorizontalFlip #采用水平反转的方式进行数据增强
    - type: Normalize #图像进行归一化
  mode: train

val_dataset: #验证数据设置
  type: Dataset #选择数据集格式
  dataset_root: data/PaddleSeg/camvid #选择数据集路径
  val_path: data/PaddleSeg/camvid/val_list.txt #选择数据集list
  num_classes: 12 #指定目标的类别个数（背景也算为一类）
  transforms: #数据预处理/增强的方式
    - type: Resize  #将原图resize成512*512在送入网络
      target_size: [512, 512]  #将原图resize成512*512在送入网络
    - type: Normalize #图像进行归一化
  mode: val

optimizer: #设定优化器的类型
  type: sgd #采用SGD（Stochastic Gradient Descent）随机梯度下降方法为优化器
  momentum: 0.9 #动量
  weight_decay: 4.0e-5 #权值衰减，使用的目的是防止过拟合

learning_rate: #设定学习率
  value: 0.1  #初始学习率
  decay:
    type: poly  #采用poly作为学习率衰减方式。
    power: 0.9  #衰减率
    end_lr: 0   #最终学习率

loss: #设定损失函数的类型
  types:
    - type: CrossEntropyLoss #损失函数类型
  coef: [1]

model: #模型说明
  type: SegNet  #设定模型类别
  num_classes: 12
```

## 七、模型信息
|  information   | description  |
|  ----  | ----  |
| Author  | chen jiejun |
| Date  | 2021.8.8 |
|Framework version|PaddlePaddle = 2.1|
|Application scenarios| Semantic Segmantation|
|Support hardware|GPU CPU|
