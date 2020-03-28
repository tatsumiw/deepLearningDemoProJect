import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import copy


def rightness(predictions,labels):
    #对于任意样本输出的第一纬度求最大，得到最大元素下标
    pred = torch.max(predictions.data,1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(labels)


#设置图像读取器的超参数
image_size = 28  #图像总尺寸28*28
num_classes = 10 #标签总类数
num_epochs = 20 #训练的总循环周期
batch_size = 64  #批处理的尺寸大小

train_dataset = dsets.MNIST('/.datas',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = dsets.MNIST('/.datas',train=False,transform=transforms.ToTensor())

#定义2个采样器
sample1 = torch.utils.data.sampler.SubsetRandomSampler(np.random.permutation(range(len(train_dataset))))
sample2 = torch.utils.data.sampler.SubsetRandomSampler(np.random.permutation(range(len(train_dataset))))
#定义2加载器
train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,sampler=sample1)
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,sampler=sample2)
#对校验数据和测试数据进行相同处理
val_size = 5000
val_indices1 = range(val_size)
val_indices2 = np.random.permutation(range(val_size))
test_indices1 = range(val_size,len(test_dataset))
test_indices2 = np.random.permutation(test_indices1)
val_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(val_indices1)
val_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(val_indices2)
test_sampler1 = torch.utils.data.sampler.SubsetRandomSampler(test_indices1)
test_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(test_indices2)

val_loader1 = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,sampler=val_sampler1)
val_loader2 = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,sampler=val_sampler2)
test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,sampler=test_sampler1)
test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,sampler=test_sampler2)

#定义网络
depth = [4,8]
class Transfer(nn.Module):
    def __init__(self):
        super(Transfer,self).__init__()
        #两个并行的卷积通道，第一个通道
        #1个输入通道，4个输出通道(4个卷积核),窗口5,填充2
        self.net1_conv1 = nn.Conv2d(1,4,5,padding=2)
        self.net1_pool = nn.MaxPool2d(2,2) #2*2池化
        #输入通道4,输出通道8(8个卷积核),窗口5,填充2
        self.net1_conv2 = nn.Conv2d(depth[0],depth[1],5,padding=2)

        #第二个通道
        #1个输入通道，4个输出通道(4个卷积核),窗口5，填充2
        self.net2_conv1 = nn.Conv2d(1,4,5,padding=2)
        #输入通道4,输出通道8(8个卷积核，窗口5,填充2)
        self.net2_covn1 = nn.Conv2d(depth[0],depth[1],5,padding=2)

        #全连接层
        self.fc1 = nn.Linear(2*image_size//4*image_size//4*depth[1],1024)
        self.fc2 = nn.Linear(1024,2*num_classes)
        self.fc3 = nn.Linear(2*num_classes,num_classes)
        self.fc4 = nn.Linear(num_classes,1)

    def forward(self, x,y,training=True):
        #网络的前馈过程,输入2张手写图像x和y，输出一个数字表示2个数字和
        #x、y都是batch_size*image_size*image_size形状的三阶向量
        #输出为batch_size长的列向量

        #首先，第1张图像进入第1个通道
        x = F.relu(self.net1_conv1(x))
        x = self.net1_pool(x)
        x = F.relu(self.net1_conv2(x))
        x = self.net1_pool(x)
        x = x.view(-1,image_size//4*image_size//4*depth[1])
        #其次，第2张图像进入第2个通道
        y = F.relu(self.net2_conv1(y))
        y = self.net1_pool(y)
        y = F.relu(self.net2_conv2(y))
        y = self.net1_pool(y)
        y = y.view(-1,image_size//4*image_size//4*depth[1])
        #将讲个卷积过来铺平的向量拼接在一齐，形成一个大向量
        z = torch.cat((x,y),1)
        z = self.fc1(z)
        z = F.relu(z)
        z = F.dropout(z,training,self.training)  #默认是0.5的概率对这层进行dropout操作
        z = self.fc2(z)
        z = F.relu(z)
        z = self.fc3(z)
        z = F.relu(z)
        z = self.fc4(z)
        return z

    #定义权重复制函数
    def set_filter_values(self,net):
        self.net1_conv1.weight.data = copy.deepcopy(net.conv1.weight.data)
        self.net1_conv1.bias.data = copy.deepcopy(net.conv1.bias.data)
        self.net1_conv2.weight.data = copy.deepcopy(net.conv2.weight.data)
        self.net1_conv2.bias.data = copy.deepcopy(net.conv2.bias.data)
        self.net2_conv1.weight.data = copy.deepcopy(net.conv1.weight.data)
        self.net2_conv1.bias.data = copy.deepcopy(net.conv1.bias.data)
        self.net2_conv2.weight.data = copy.deepcopy(net.conv2.weight.data)
        self.net2_conv2.bias.data = copy.deepcopy(net.conv2.bias.data)

#加载原手写识别模型，需重新定义源网络的类
depth = [4,8]
class ConvNet(nn.Module):
    def __init__(self):
        #首先调用父类相应的构造函数
        super(ConvNet,self).__init__()

    def forward(self, x):
        #完成前向运算，各组件的实际拼装
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,image_size//4*image_size//4*depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

original_net = torch.load('minst_conv_checkpoint') #加载模型
net = Transfer() #构造网络
net.set_filter_values(original_net) #数据迁移
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr=0.0001,momentum=0.9) #加载参数至优化器中
 
#测试与训练
records = []
for epoch in range(num_epochs):
    losses = []
    for idx,data in enumerate(train_loader1,train_loader2):
        ((x1, y1), (x2, y2)) = data
        optimizer.zero_grad()
        net.train()
        outputs = net(x1,x2)
        labels = y1 + y2
        loss = criterion(outputs, labels.type(torch.dtype))
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())
        if idx%100==0:
            val_losses = []
            rights = []
            net.eval()
            for val_data in zip(val_loader1,val_loader2):
                ((x1,y1),(x2,y2)) = val_data
                outputs = net(x1,x2)
                labels = y1 + y2
                loss = criterion(outputs,labels.type(torch.dtype))
                val_losses.append(loss,data.item())

                right = rightness(outputs.data,labels)
                rights.append(right)
                right_ratio = 1.0* np.sum([i[0] for i in rights])/np.sum([i[1] for i in rights])
                print('第{}周期，第{}/{}个批，训练误差:{}、校验误差:{:.2f}、准确率:{:.2f}'.format(
                    epoch,idx,len(train_loader1),np.mean(losses),np.mean(val_losses),right_ratio
                ))
            records.append([np.mean(losses),np.mean(val_losses),right_ratio])
