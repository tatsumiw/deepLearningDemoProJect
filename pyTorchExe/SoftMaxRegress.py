
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import DiveIntoPytorchUtils as utils
from collections import OrderedDict


"""
    使用Pytorch实现softmax回归模型
"""

#获取数据集
#ToTensor 将尺寸为C*H*W数据为0-255的PIL图片或者数据类型为np.uint8的Numpy数组转为尺寸为C*H*W且数据类型为torch.float32且0-1的Tensor
mnist_train = torchvision.datasets.FashionMNIST('../datas/Datasets/FashionMNist',train=True,transform =transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST('../datas/Datasets/FashionMNist',train=False,transform =transforms.ToTensor())
print(len(mnist_train),len(mnist_test))   #查看样本数量
#查看某条样本
feature,label = mnist_train[0]
print(feature.shape,label)   #C*H*W
#查看样本的图像内容 和文本标签
X,y = [],[]
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
utils.show_fashion_mnist(X,utils.get_fashion_mnist_labels(y))

#读取小批量数据
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0 #0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True,num_workers=num_workers)

#模型定义
num_inputs = feature.shape[1] * feature.shape[2] #输入神经元数量为样本的W*H
num_hiddens1, num_hiddens2 =  256, 256           #中间层神经元个数
num_outputs = 10                                 #输出神经元数量为10

#展开层(将图像H*W压缩至1维)
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):   #shape: (batch, 1, 28, 28)
        return x.view(x.shape[0],-1) #展开成 shape(batch,1*28*28)
#模型组合
drop_prob1, drop_prob2 = 0.2, 0.5 #设置dropout接近输入层的丢弃概率稍小
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1,num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, num_outputs)
)
#初始化参数
for params in net.parameters():
    init.normal_(params,mean=0,std=0.01)
lossfc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

#模型训练
num_epochs = 5
for epoch in range(num_epochs):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        loss = lossfc(y_hat, y).sum()
        optimizer.zero_grad() # 梯度清零
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    test_acc = utils.evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_loss_sum/n, train_acc_sum/n, test_acc))

