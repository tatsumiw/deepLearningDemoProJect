from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy


#图像加载函数
def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

#绘制图像的函数
def imshow(tensor,title=None):
    image = tensor.clone().cpu()
    image = image.view(3,imsize,imsize)  #删除添加的batch层
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(1) #展示图片时间

#定义Gram矩阵
def Gram(input):
    a,b,c,d = input.size() #a=batch size(=1) b=特征图数量 (c,d)=特征图的图像尺寸
    features = input.view(a*b,c*d) #扁平化操作
    G = torch.mm(features,features.t())
    return G.div(a*b*c*d)

style = 'images/escher.jpg'       #风格图像路径(大小需一致)
content = 'images/portrait2.jpg'  #内容图像路径(大小需一致)
style_weight = 1000               #风格损失所占比重
content_weight = 1                #内容损失所占比重
imsize = 128                      #希望得到的图像大小(越大越清晰)

loader = transforms.Compose([transforms.Scale(imsize),transforms.ToTensor()])  #加载图像为指定大小,并转为Tensor
#载入图片
style_img = image_loader(style).type(torch.FloatTensor)
content_img = image_loader(content).type(torch.FloatTensor)
unloader = transforms.ToPILImage()
plt.ion()
plt.figure()
imshow(style_img.data,'Style Image')
plt.figure()
imshow(content_img.data,'Contemt Image')

#加载VGG
cnn = models.vgg19(pretrained=True).features
#定义内容损失和风格损失模块
class ContentLoss(nn.Module):
    def __init__(self,target,weight):
        super(ContentLoss,self).__init__()
        self.target = target.detach()*weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input*self.weight,self.target)
        self.output = input
        return self.output

    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class StyleLoss(nn.Module):
    def __init__(self,target,weight):
        super(StyleLoss,self).__init__()
        self.target = target.detach()*weight
        self.weight = weight
        # self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self_G = Gram(input)
        self_G.mul_(self.weight)
        self.loss = self.criterion(self_G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

#设置需要的层
content_layers = ['conv_4'] #内容损失
style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5'] #风格损失
#定义列表存储每一个周期的计算损失
content_losses = []
style_losses = []
model = nn.Sequential()
#循环VGG的每一层,同时加入风格计算层和内容计算层,构造一个全新的神经网络
#将每层卷积核的数据都加载到新的网络模型上
i = 1
for layer in list(cnn):
    if isinstance(layer,nn.Conv2d):
        name = 'conv_' + str(i)
        model.add_module(name,layer)

        if name in content_layers:
            #如果当前层模型位于定义好的要计算的内容的层
            target = model(content_img).clone()
            content_loss = ContentLoss(target,content_weight)
            model.add_module("content_loss_"+str(i),content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 如果当前层模型位于定义好的要计算的风格的层
            target_feature = model(style_img).clone()
            target_feature_gram = Gram(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer,nn.ReLU):
        #如果不是卷积层
        name = "relu_" + str(i)
        model.add_module(name,layer)
        i+=1

    if isinstance(layer,nn.MaxPool2d):
        name = "pool_" + str(i)
        model.add_module(name,layer)

#风格训练
input_img = torch.randn(content_img.data.size()).type(torch.FloatTensor) #输入原始图，生成随机噪声图
plt.figure()
imshow(input_img.data,title='Input Image')
#迭代训练
input_param = nn.Parameter(input_img.data)
optimizer = optim.LBFGS([input_param]) #擅长计算大规模数据梯度下降
num_steps = 300 #迭代步数

print("正在构造风格迁移模型...")
print("开始优化...")
for i in range(num_steps):
    input_param.data.clamp_(0,1) #限制输入的图像色彩数据范围在0-1之间
    optimizer.zero_grad()
    model(input_param)
    style_score = 0
    content_score = 0

    #每个损失函数都开始反向传播算法
    for sl in style_losses:
        style_score += sl.backward()
    for cl in content_losses:
        content_score += cl.backward()
    if i%2==0:
        print('运行{}轮,风格损失:{:4f},内容损失:{:4f}'.format(i,style_score.data.item(),content_score.data.item()))

    optimizer.step(lambda :style_score + content_score)

#做修正，防止数据超界
output = input_param.data.clamp_(0,1)
#打印结果
plt.figure()
imshow(output,title='Output Image')
plt.ioff()
plt.show()