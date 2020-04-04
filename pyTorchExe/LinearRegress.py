import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


"""
    使用Pytorch实现简单线性回归模型
"""

#线性回归实现
#生成数据集
num_input = 2
num_examples = 1000
true_w = [2,-3,4]
true_b = 4.2
features = torch.randn(num_examples,num_input,dtype=torch.float32)      #生成特征
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b #生成标签
labels = labels + torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32) #加上随机干扰
print(features[0],labels[0])     #查看第一条样本数据形式

#读取数据
batch_size = 10  #batch数量
dataset = Data.TensorDataset(features,labels)  #特征和标签组合成数据集
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True) #随机读取最小批量

#定义模型
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_feature,1)  #线性层神经元输入输出数量

    def forward(self,x):
        y = self.linear(x)  #前向传播逻辑
        return y

net = LinearNet(num_input)
print(net)

#初始化模型参数
init.normal_(net.linear.weight,mean=0,std=0.01)
init.constant_(net.linear.bias,val=0)
#定义损失函数
lossfc = nn.MSELoss()
#定义优化算法
optimizer = optim.SGD(net.parameters(),lr=0.03)
print(optimizer)

#训练模型
num_epochs = 30
for epoch in range(1,num_epochs+1):
    for X, y in data_iter:
        output = net(X)
        loss = lossfc(output,y.view(-1,1)) #y转换成1列,计算loss loss是一个标量
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, loss.item()))


#查看模型学习到的参数，对比我们之前定义的w和b
dense = net.linear
print(true_w,'->',dense.weight.data.numpy()[0])
print(true_b,'->',dense.bias.data.numpy()[0])