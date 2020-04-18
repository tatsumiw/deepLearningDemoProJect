import numpy as np
import torch
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

#c超参数定义
train_batch_size = 64  #每次训练的样本量
test_batch_size = 128  #每次测试的样本量
num_epoches = 20       #模型数据迭代次数
lr = 0.01
momentum = 0.5

#下载数据并对数据进行预处理(数据维度 c=1 h*w=28*28)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]) #均值方差归一化
train_dataset = mnist.MNIST('./data',train=True,transform=transform,download=True)
test_dataset = mnist.MNIST('./data',train=False,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=test_batch_size,shuffle=False)

#构建网络
class Net(nn.Module):
    """
    使用sequential构建网络
    """
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
#实例化网络
model = Net(28*28,300,100,10)
#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum)

#训练模型
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()

    #动态修改模型参数学习率
    if epoch%5==0:
        optimizer.param_groups[0]['lr']*=0.1 #能拿到optimizer的参数
    for img,label in train_loader:
        img = img.view(img.shape[0],-1)      #数据输入处理(把batch*1*28*28展开成batch*784)
        out = model(img)                     #输出预测结果(batch*out_dim)
        loss = criterion(out,label)          #算出在每个baych中的loss是个value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #记录误差
        train_loss += loss.item()
        #计算准确率
        _,pred = out.max(1)                  #取出每个样本(维度1)最大的值 前部分为具体得分值 后部分为对应的label
        num_correct = (pred==label).sum().item() #pred和label都是batch*1的tensor
        acc = num_correct/img.shape[0]
        train_acc += acc

        break
    losses.append(train_loss/len(train_loader))
    acces.append(train_acc/len(train_loader))

    # #预测模式
    # eval_loss = 0
    # eval_acc = 0
    # model.eval()
    # for img,label in test_loader:
    #     img = img.view(img.shape[0],-1)
    #     out = model(img)
    #     loss = criterion(out,label)
    #     #记录误差
    #     eval_loss += loss.item()
    #     #记录准确率
    #     _,pred = out.max(1)
    #     num_correct = (pred==label).sum().item()
    #     acc = num_correct/img.shape[0]
    #     eval_acc += acc
    # eval_losses.append(eval_loss/len(test_loader))
    # eval_acces.append(eval_acc/len(test_loader))

    #整体结果输出
    print('epoch:{},TrainLoss:{:.4f},TrainAcc:{:.4f},TestLoss:{:.4f},TestAcc:{:.4f}'.format(
        epoch,train_loss/len(train_loader),train_acc/len(train_loader),eval_loss/len(test_loader),eval_acc/len(test_loader)
    ))

    break