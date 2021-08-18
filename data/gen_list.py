#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# 创建list
import os

dir = "camvid"
# %cd camvid
# !touch train_list.txt
# !touch val_list.txt
# %cd
os.mkdir("train_list.txt")
os.mkdir("val_list.txt")
names = os.listdir(os.path.join(dir, "train"))
l = len(names)
print("总共%d个数据" %l)
n = 0
for name in names:
    with open("camvid/train_list.txt","r+") as f:
        f.read()
        # t = "camvid/train/" + name + " camvid/trainannot/" + name + "\n"
        t = "train/" + name + " trainannot/" + name + "\n"
        f.write(t)
    n+=1
    # with open("train_list.txt","r+") as f:
    #     f.read()
    #     t = "camvid/train/" + name + " camvid/trainannot/" + name + "\n"
    #     f.write(t)
    # n+=1
    
print("已写入%d路径" %n)

dir = "camvid"
names = os.listdir(os.path.join(dir, "val"))
l = len(names)
print("总共%d个数据" %l)
n = 0
for name in names:
    with open("camvid/val_list.txt","r+") as f:
        f.read()
        # t = "camvid/val/" + name + " camvid/valannot/" + name + "\n"
        t = "val/" + name + " valannot/" + name + "\n"
        f.write(t)
    n+=1
    # with open("val_list.txt","r+") as f:
    #     f.read()
    #     t = "camvid/val/" + name + " camvid/valannot/" + name + "\n"
    #     f.write(t)
    # n+=1
    
print("已写入%d路径" %n)

dir = "camvid"
# %cd camvid
# !touch train__list.txt
# !touch val_list.txt
# %cd
os.mkdir("train__list.txt")
names = os.listdir(os.path.join(dir, "train"))
l = len(names)
print("总共%d个数据" %l)
n = 0
for name in names:
    with open("train__list.txt","r+") as f:
        f.read()
        t = "camvid/train/" + name + " camvid/trainannot/" + name + "\n"
        f.write(t)
    n+=1
    
print("已写入%d路径" %n)