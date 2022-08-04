# -*- coding: utf-8 -*-
"""
torch中softmax接口的使用，训练时softmax常与交叉熵损失函数配套使用
torch中有三种常用的softmax接口，分别为
torch.nn.Softmax() #最常用
torch.nn.Softmax2d() #图片softmax
torch.nn.LogSoftmax() #softmax取对数，常与NLLLoss联合使用实现交叉熵损失计算
"""

import torch

"""
生成模拟数据
"""
logits = torch.autograd.Variable(torch.tensor([[2,  0.5,6], [0.1,0,  3]]))
labels = torch.autograd.Variable(torch.LongTensor([2,1]))
print(logits)
print(labels)

"""
算softmax
"""
print('Softmax:',torch.nn.Softmax(dim=1)(logits))


"""
torch.nn.NLLLOSS通常不被独立当作损失函数，而需要和softmax、log等运算组合当作损失函数。
"""
logsoftmax = torch.nn.LogSoftmax(dim=1)(logits)
print('logsoftmax:',logsoftmax)
output = torch.nn.NLLLoss()(logsoftmax, labels)
print('NLLLoss:',output)


"""
torch.nn.CrossEntropyLoss相当于softmax + log + nllloss。
"""
print ( 'CrossEntropyLoss:', torch.nn.CrossEntropyLoss()(logits, labels) )