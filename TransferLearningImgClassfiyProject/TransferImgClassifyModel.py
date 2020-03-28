import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import copy
import os

#定义计算分类准确度的函数
def rightness(predictions,labels):
    #对于任意样本输出的第一纬度求最大，得到最大元素下标
    pred = torch.max(predictions.data,1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(labels)

if __name__ == '__main__':

    #数据存储总路径
    data_dir = 'data'
    #图像的大小为224*224像素
    image_size = 224
    #从data_dir/train加载文件
    #加载的过程将会对图像进行如下图像增强操作
    #1.随机从原始图像中切下来一块224*224大小的区域
    #2.随机水平翻转图像
    #3.将图像的色彩数据标准化
    train_dataset = datasets.ImageFolder(os.path.join(data_dir,'train'),transforms.Compose([
        transforms.RandomSizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]))
    #加载验证数据集，对每个加载的数据进行如下处理
    #1.放大到256*256像素
    #2.从中心区域切割下224*224大小的图像区域
    #3.将图像的色彩数据标准化
    val_dataset = datasets.ImageFolder(os.path.join(data_dir,'val'),transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]))
    #创建对应的数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=4)
    validation_loader = torch.utils.data.DataLoader(val_dataset,batch_size=4,shuffle=True,num_workers=4)
    #读取数据中的分类类别数
    num_classes = len(train_dataset.classes)

    #模型迁移
    #预训练方式
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features    #resNet最后全连接层输入神经元数目
    net.fc = nn.Linear(num_ftrs,2)   #输出为2类
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
    #固定值方式训练
    # net = models.resnet18(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False  #把梯度的属性都更新为false
    # num_ftrs = net.fc.in_features    #resNet最后全连接层输入神经元数目
    # net.fc = nn.Linear(num_ftrs,2)   #输出为2类
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)

    #训练模型
    record = []
    num_epochs = 20
    net.train(True) #给网络做标记，说明模型在训练集上训练
    for epoch in range(num_epochs):
        train_rights = []
        train_losses = []
        for batch_idx,(data,target) in enumerate(train_loader):
            data,target = data,target
            output = net(data)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #计算准确率
            right = rightness(output,target)
            train_rights.append(right)
            train_losses.append(loss.data.item())

        print(np.mean(train_losses),np.mean(train_rights))
