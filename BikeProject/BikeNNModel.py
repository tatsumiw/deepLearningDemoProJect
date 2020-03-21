"""
使用人工神经网络预测自行车使用量
"""

import numpy as np
import pandas as pd
import torch
# from torch.autograd import Variable

#读取数据
rides = pd.read_csv(".\hour.csv")
rides.head()

#类型型变量处理(one-hot)
dummy_fields = ['season','weathersit','mnth','hr','weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each],prefix=each,drop_first=False) #one-hot处理
    rides = pd.concat([rides,dummies],axis=1)
#删除原来的类别型变量
fields = ["instant","dteday","season","weathersit","weekday","atemp","mnth","workingday","hr","casual","registered"]
data = rides.drop(fields,axis=1)
#连续型变量处理(归一化)
quant_features = ['cnt','temp','hum','windspeed']
for each in quant_features:
    mean, std = data[each].mean(),data[each].std()
    data[each] = data[each].apply(lambda x : (x-mean)/std)
#数据集划分
train_data = data[:-21*24]
test_data = data[-21*24:]
target_fields = ['cnt']
features,target = train_data.drop(target_fields,axis=1),train_data[target_fields]
test_features,test_target = test_data.drop(target_fields,axis=1),test_data[target_fields]
#数据转化为Numpy
X = features.values
Y = target.values.astype(float)

#构建神经网络
input_size = features.shape[1] #输入层神经元个数
hidden_size = 10
output_size = 1
batch_size = 128
neu = torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,output_size)
)
cost = torch.nn.MSELoss() #损失函数
optimizer = torch.optim.SGD(neu.parameters(),lr=0.01)

#模型训练(数据重复训练1000次)
losses = []
for i in range(1000):
    batch_loss = []
    for start in range(0,len(X),batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = torch.FloatTensor(X[start:end])
        yy =torch.FloatTensor(Y[start:end])
        predict = neu(xx)
        loss = cost(predict,yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
        if i%100 == 0:
            losses.append(np.mean(batch_loss))
            print(i,np.mean(batch_loss))

#对数据进行预测
targets = test_target['cnt'].values.astype(float)
x = torch.FloatTensor(test_features.values)
y = torch.FloatTensor(test_target.values)
predict = neu(x).data.numpy()
print(predict)
print(predict.shape)
