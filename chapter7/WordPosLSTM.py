"""
使用LSTM进行词性判断
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


#定义训练数据
train_data = [
    (("The cat ate the fish".split(),["DET","NN","V","DET","NN"])),
    (("They read that book").split(),["NN","V","DET","NN"])
]
#定义测试数据
testing_data = [("They ate the fish".split())]

#构建每个单词的索引字典
word_to_ix = {}
for sent,tags in train_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

#手工设置词性的索引字典
tag_to_idx = {"DET":0,"NN":1,"V":2}
print(word_to_ix)
print(tag_to_idx)

#转换数据为模型的Tensor形式
def prepare_sequence(seq,to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return tensor
#
#构建网络
class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size):
        super(LSTMTagger,self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim,tagset_size)

    #初始化隐含状态State及C
    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim))

    def forward(self,sentence):
        #获得词嵌入矩阵
        embeds = self.word_embeddings(sentence)  #每个单词从1纬度转为embedding_dim维度
        #按lstm格式，修改embeds的形状(需要输入训练数据和隐层数目),返回输出和隐层
        #训练数据为 序列长度:句子单词数,batch,单词维度
        lstm_out,self.hidden = self.lstm(embeds.view(len(sentence),1,-1),self.hidden) #(num_layers*num_direstions,batch,hidden_size)
        #修改隐含状态的形状，作为全连接层的输入
        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        tag_scores = F.log_softmax(tag_space,dim=1)
        return tag_scores


#训练网络
EMBEDDING_DIM = 10
HIDDEN_DIM = 3 #这里等于词性个数
model = LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_ix),len(tag_to_idx))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(400):
    for sentence,tag in train_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence,word_to_ix)  #句子单词索引后的列表
        targets = prepare_sequence(tag,tag_to_idx)           #句子单词词性索引后的列表
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores,targets)
        loss.backward()
        optimizer.step()


#查看模型训练的结果
print(train_data[0][0])
inputs = prepare_sequence(train_data[0][0],word_to_ix)
print(inputs)
tag_scores = model(inputs)
print(tag_scores)
print(torch.max(tag_scores,1))


#测试模型
print(testing_data[0])
test_input = prepare_sequence(testing_data[0],word_to_ix)
print(test_input)
tag_scores01 = model(test_input)
print(tag_scores01)
print(torch.max(tag_scores01,1))