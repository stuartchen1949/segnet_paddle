# 需要安装PaddleSeg，并且从Github clone PaddleSeg的代码，使用自定义config文件覆盖他们

import paddle
import os

# 预备PaddleSeg环境
os.system("git clone https://gitee.com/paddlepaddle/PaddleSeg.git && pip install PaddleSeg")

# 覆盖配置文件（主要是 loss 计算文件截止提交时有bug）
os.system("cp cross_entropy_loss.py PaddleSeg/paddleseg/models/losses/")

