from torchsummary import summary
from torchvision.models import vgg16  # 以 vgg16 为例
import torch

path = '/Users/zhangwenyou/PycharmProjects/AIstudy4/minGPT/projects/adder/out/adder/model.pt'
pretrained_dict = torch.load(path)

myNet = vgg16()  # 实例化网络，可以换成自己的网络
# myNet = torch.load(path)
summary(myNet, (3, 64, 64))  # 输出网络结构

for k, v in pretrained_dict.items():  # k 参数名 v 对应参数值
    print('k==========', k)
    print('v===================', v)
