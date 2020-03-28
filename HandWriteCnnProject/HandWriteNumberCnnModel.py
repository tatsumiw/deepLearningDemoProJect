import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

def rightness(predictions,labels):
    #对于任意样本输出的第一纬度求最大，得到最大元素下标
    pred = torch.max(predictions.data,1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(labels)


#定义训练超参数
image_size = 28  #图像总尺寸大小为28*28
num_classes = 10 #标签总类数
num_epochs = 10  #训练的总循环周期
batch_size = 64  #一个批次的大小，64张图片

#数据加载
train_datset = dsets.MNIST('./data',train=True,transform=transforms.ToTensor(),download=True) #训练集60000
test_datset = dsets.MNIST('./data',train=False,transform=transforms.ToTensor())               #测试集10000

#数据加载器(训练集，验证集，测试集)
train_loader = torch.utils.data.DataLoader(dataset=train_datset,batch_size=batch_size,shuffle=True)
indices = range(len(test_datset))
indices_val = indices[:5000]   #测试集中划分0-5000为验证集
indices_test = indices[5000:]  #c测试集中划分5000-10000为测试集
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)   #定义抽样方法
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test) #定义抽样方法
validation_loader = torch.utils.data.DataLoader(dataset=test_datset,batch_size=batch_size,shuffle=False,sampler=sampler_val)
test_loader = torch.utils.data.DataLoader(dataset=test_datset,batch_size=batch_size,shuffle=False,sampler=sampler_test)

#构建模型
#定义卷积神经网络,4和8为人为指定的2个卷积层的厚度
depth = [4,8]
class ConvNet(nn.Module):
    def __init__(self):
        #首先调用父类相应的构造函数
        super(ConvNet,self).__init__()
        #定义一个卷积层，输入通道为1，输出通道为4，窗口大小为5，padding为2
        self.conv1 = nn.Conv2d(1,4,5,padding=2)
        self.pool = nn.MaxPool2d(2,2)
        #定义2层卷积，输入通道为depth[0]，输出通道为depth[1],窗口为5，padding为2
        self.conv2 = nn.Conv2d(depth[0],depth[1],5,padding=2)
        #一个线性连接层，输入尺寸为一嘴个一层立方体的线性平铺,输出层512个节点
        self.fc1 = nn.Linear(image_size//4*image_size//4*depth[1],512)
        self.fc2 = nn.Linear(512,num_classes)

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

    def retrieve_features(self,x):
        #该函数用于保存提取的特征图
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1,feature_map2)

#运行模型
net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

record = []  #记录准确率等数值
weights = [] #没若干不记录一次卷积核

#开始循环训练
for epoch in range(num_epochs):
    train_rights = [] #记录训练数据集准确率的容器

    #训练每次拿64个训练数据训练
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data, target
        net.train()
        output = net(data)  # 完成一次前向计算
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)

        if batch_idx % 1 == 0:
            net.eval()
            val_rights = []
            # 开始在验证集上做循环，计算验证集上的准确率
            for(data,target) in validation_loader:
                data,target = data,target
                output = net(data)
                right = rightness(output,target)
                val_rights.append(right)
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
            print('训练周期:{}[{}/{}({:.0f}%)],Loss:{:.6f},训练准确率:{:.2f}%,校正准确率:{:.2f}%'.format(
                epoch,
                batch_idx*len(data),
                len(train_loader.dataset),
                100.*batch_idx/len(train_loader),
                loss.data,
                100.*train_r[0]/train_r[1],
                100.*val_r[0]/val_r[1]
            ))

            record.append((100-100.*train_r[0]/train_r[1],100-100.*val_r[0]/val_r[1]))
            weights.append([net.conv1.weight.data.clone(),net.conv1.bias.data.clone(),
                            net.conv2.weight.data.clone(),net.conv2.bias.data.clone()])

#测试模型
net.eval()
vals = []
for data,target in test_loader:
    data,target = data,target
    output = net(data)
    val = rightness(output,target)
    vals .append(val)

#计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 * rights[0]/rights[1]
right_rate

plt.figure(figsize=(10,7))
plt.plot(record)
plt.xlabel('step')
plt.ylabel('error rate')

#保存模型
torch.save(net,'minst_conv_checkpoint')