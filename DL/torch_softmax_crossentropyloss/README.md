关键词：pytorch、softmax、cross entropy loss
torch中softmax接口的使用，训练时softmax常与交叉熵损失函数配套使用
torch中有三种常用的softmax接口，分别为
torch.nn.Softmax() #最常用
torch.nn.Softmax2d() #图片softmax
torch.nn.LogSoftmax() #softmax取对数，常与NLLLoss联合使用实现交叉熵损失计算