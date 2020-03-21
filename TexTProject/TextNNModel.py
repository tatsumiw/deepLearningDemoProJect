import numpy as np
import torch
import torch.nn as nn
import re
import jieba
from collections import Counter
import math

#对文本的标点符号过滤
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）;\[\]]+","",sentence)
    return sentence

#扫描所有文本，分词并建立词典，分出正向的负向的评论
def prepare_data(good_file,bad_file,is_filter=True):
    all_words = [] #存储所有的单词
    pos_sentence = [] #存储正向的评论
    neg_sentence = [] #存储负向的评论
    with open(good_file,'r',encoding='utf-8') as f:
        for idx,line in enumerate(f):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line) #分词
            if(len(words)) >0:
                all_words += words
                pos_sentence.append(words)
    print('{0}包含{1}行，{2}个词'.format(good_file,idx+1,len(all_words)))
    count = len(all_words)
    with open(bad_file,'r',encoding='utf-8') as f:
        for idx,line in enumerate(f):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if(len(words)) >0:
                all_words += words
                neg_sentence.append(words)
    print('{0}包含{1}行，{2}个词'.format(bad_file, idx + 1, len(all_words)))

    #建立词典 diction的每一行为{w:[id,出现次数]}
    diction = {}
    cnt = Counter(all_words)
    for word ,freq in cnt.items():
        diction[word] = [len(diction),freq]
    print('字典大小：{}'.format(len(diction)))
    return (pos_sentence,neg_sentence,diction)

#根据单词返回单词编码
def word2index(word,diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return value

#根据编码找到对应的单词
def index2word(index,diction):
    for w,v in diction.items():
        if v[0] == index:
            return w
    return None

#输入一个句子和对应的词典，得到句子的向量化表示
#向量的大小为词典中词汇个数
def sentence2vec(sentence,diciionary):
    vector = np.zeros(len(diciionary))
    for i in sentence:
        vector[i] += 1
    return 1.0*vector/len(sentence)

#定义计算分类准确度的函数
def rightness(predictions,labels):
    #对于任意样本输出的第一纬度求最大，得到最大元素下标
    pred = torch.max(predictions.data,1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(labels)



if __name__ == '__main__':
    good_file = './good.txt'
    bad_file = './bad.txt'
    #完成数据
    pos_sentence,neg_sentence,diction = prepare_data(good_file,bad_file,True)
    # st = sorted([(v[1],w)for w,v in diction.items()])

    #遍历所有句子，将每一个词映射成编码
    dataset = []
    labels = []
    sentences = []
    #处理正向评论(生成8052条正样本，每条正样本7156维向量)
    for sentence in pos_sentence:
        new_sentence = []
        for l in sentence:
            if l in diction:
                new_sentence.append(word2index(l,diction))
        dataset.append(sentence2vec(new_sentence,diction))
        labels.append(0)
        sentences.append(sentence)

    #处理负向评论(生成4983条正样本，每条负样本7156维向量)
    for sentence in neg_sentence:
        new_sentence = []
        for l in sentence:
            if l in diction:
                new_sentence.append(word2index(l,diction))
        dataset.append(sentence2vec(new_sentence,diction))
        labels.append(1)
        sentences.append(sentence)

    #打乱所有的数据顺序，形成数据集
    indices = np.random.permutation(len(dataset)) #打乱数据顺序
    #根据打乱的下标，重新生成数据集，标签集，以及对应的原始句子
    dataset = [dataset[i] for i in indices]
    labels = [labels[i] for i in indices]
    sentences = [sentences[i] for i in indices]
    #将整个数据集划分为训练集、校验集和测试集
    test_size = len(dataset)
    train_data = dataset[:math.floor(0.8*test_size)]
    train_label = labels[:math.floor(0.8*test_size)]
    valid_data = dataset[math.floor(0.8*test_size):math.floor(0.9*test_size)]
    valic_label = labels[math.floor(0.8*test_size):math.floor(0.9*test_size)]
    test_data = dataset[math.floor(0.9*test_size):]
    test_label = labels[math.floor(0.9*test_size):]

    #构建神经网络
    #第一层是线性层加RELU,第二层为线性层，中间有10个神经元(7156,10,2)
    model = nn.Sequential(
        nn.Linear(len(diction),10),
        nn.ReLU(),
        nn.Linear(10,2),
        nn.LogSoftmax(dim=1)
    )
    #损失函数的交叉熵
    cost = torch.nn.NLLLoss()
    #优化算法为SGD
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    records = []
    #循环10个epoch
    losses = []
    for epoch in range(10):
        for i,data in enumerate(zip(train_data,train_label)):
            x,y = data
            x = torch.FloatTensor(x).view(1,-1)
            y = torch.LongTensor(np.array([y]))
            optimizer.zero_grad()
            predict = model(x)
            loss = cost(predict,y)
            losses.append(loss.data.item())
            loss.backward()
            optimizer.step()

            if i%1 == 0:
                val_losses = []
                rights = []
                for j ,val in enumerate(zip(valid_data,valic_label)):
                    x,y = val
                    x = torch.FloatTensor(x).view(1,-1)
                    y = torch.LongTensor(np.array([y]))
                    predict = model(x)
                    right = rightness(predict,y)
                    rights.append(right)
                    loss = cost(predict,y)
                    val_losses.append(loss.data.item())

                #计算验证集的效果
                right_ratio = 1.0*np.sum([i[0] for i in rights])/np.sum([i[1] for i in rights])
                print('第{}轮，训练损失：{:.2f}，校验损失:{:.2f},校验准确率:{:.2f}'
                      .format(epoch,np.mean(losses),np.mean(val_losses),right_ratio))
                records.append([np.mean(losses),np.mean(val_losses),right_ratio])
