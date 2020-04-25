import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# 下载数据(图像大小是 c*h*w:3*32*32)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
trainset = torchvision.datasets.CIFAR10('./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 随机查看部分数据
# dataiter = iter(trainloader)
# images,label = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# print(images)
# print(images.shape)
# print(label)
# print(' '.join('%5s'%classes[label[j]] for j in range(4)))

#构建网络模型
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool2d(1) #使用全局池化层
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape) x.shape = 4*36*6*6
        x = self.aap(x)
        x = x.view(x.shape[0],-1)
        x = self.fc3(x)
        return x


# 模型实例化
net = CNNNet()
print(net)

# 取模型前四层
# nn.Sequential(*list(net.children())[:4])
# # 初始化参数
# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         # nn.init.normal_(m.weight)
#         # nn.init.xavier_normal_(m.weight)
#         nn.init.kaiming_normal_(m.weight)  # 卷积层参数初始化
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight)  # 全连接层参数初始化

# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  #可增加训练次数
    running_loss = 0.0
    for i,data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        #显示损失值
        running_loss += loss.item()
        if i%2000==1999: #print every 2000 mini-batches
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

#测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs ,labels = data
        outpus = net(inputs)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of thr network on the 10000 test images: %d %%' %(100*correct/total)) #测试集下总体效果
