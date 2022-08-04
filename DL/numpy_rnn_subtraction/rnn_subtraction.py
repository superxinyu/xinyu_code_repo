"""
用numpy模拟RNN对两个二进制数字进行相减
numpy模拟RNN中输入层、隐层、输出层的构造，并通过正向传播计算损失，通过反向传播计算梯度并更新权重
数据就是两个数字的二进制相减，每一位的两个数字作为输入，输出为该位置对应的相减结果
因为每一位的结果不仅依赖于当前位的输出，还严重依赖于前一位是否借位，所以用RNN模拟
"""

import copy, numpy as np
np.random.seed(0) #随机数生成器的种子，可以每次得到一样的值
# compute sigmoid nonlinearity
def sigmoid(x): #激活函数
    output = 1/(1+np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):#激活函数的导数
    return output*(1-output)


"""
建立二进制映射，最大数设置为2^8=256，用二进制是因为输入为1或0，符合输入的形式
"""
int2binary = {} #整数到其二进制表示的映射
binary_dim = 8 #暂时制作256以内的减法
## 计算0-256的二进制表示
largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)  # unpackbits函数可以把整数数组转化成2进制数。
for i in range(largest_number):
    int2binary[i] = binary[i]


"""
定义参数
"""
# input variables
alpha = 0.9 #学习速率
input_dim = 2  # 输入的维度是2，因为输入是两个数的某一位上的两个数字（0或1），所以是2
hidden_dim = 16  # 隐层自定义设置即可
output_dim = 1  # 输出维度为1，即输入对应的两个数相减的结果（并不是单纯的相减，结果依赖于前一位是否借位）

# initialize neural network weights
synapse_0 = (2*np.random.random((input_dim,hidden_dim)) - 1)*0.05  # 隐藏层权重。。维度为2*16， 2是输入维度，16是隐藏层维度
synapse_1 = (2*np.random.random((hidden_dim,output_dim)) - 1)*0.05  # 循环节点权重
synapse_h = (2*np.random.random((hidden_dim,hidden_dim)) - 1)*0.05  # 输出层权重(写反了吧？这个应该是循环节点权重？)
# => [-0.05, 0.05)，

# 用于存放反向传播的权重更新值，为简化代码，这里先不设置偏置
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)



# training 
for j in range(10000):

    """
    随机生成被减数与减数a、b，并转为二进制
    """
    #生成一个数字a
    a_int = np.random.randint(largest_number) 
    #生成一个数字b,b的最大值取的是largest_number/2,作为被减数，让它小一点。
    b_int = np.random.randint(largest_number/2) 
    #如果生成的b大了，那么交换一下
    if a_int<b_int:
        tt = a_int
        b_int = a_int
        a_int=tt
    
    a = int2binary[a_int] # binary encoding
    b = int2binary[b_int] # binary encoding    
    # true answer
    c_int = a_int - b_int
    c = int2binary[c_int]
    
    # 存储神经网络的预测值
    d = np.zeros_like(c)
    overallError = 0 #每次把总误差清零
    
    layer_2_deltas = list() #存储每个时间点输出层的误差
    layer_1_values = list() #存储每个时间点隐藏层的值
    
    layer_1_values.append(np.ones(hidden_dim)*0.1) # 一开始没有隐藏层，所以初始化一下原始值为0.1

    """
    正向传播
    """
    # moving along the positions in the binary encoding
    for position in range(binary_dim):#循环遍历每一个二进制位
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])#从右到左，每次去两个输入数字的一个bit位
        y = np.array([[c[binary_dim - position - 1]]]).T#正确答案
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))  # （输入层 + 之前的隐藏层） -> 新的隐藏层，这是体现循环神经网络的最核心的地方！！！通过隐层传递借位信息
        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1)) #隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层
        
        layer_2_error = y - layer_2  # 误差
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2)) #把每一个时间点的误差导数都记录下来
        overallError += np.abs(layer_2_error[0])#总误差
    
        d[binary_dim - position - 1] = np.round(layer_2[0][0]) #记录下每一个预测bit位
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))#记录下隐藏层的值，在下一个时间点用
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    #反向传播，从最后一个时间点到第一个时间点
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]]) #最后一次的两个输入
        layer_1 = layer_1_values[-position-1] #当前时间点的隐藏层,因为layer_1_values列表是先加的后位，所以-1位置就是前位
        prev_layer_1 = layer_1_values[-position-2] #前一个时间点（下一位）的隐藏层
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1] #当前时间点输出层导数
        # error at hidden layer
        # 通过后一个时间点（因为是反向传播）的隐藏层误差和当前时间点的输出层误差，计算当前时间点的隐藏层误差
        # 即 loss= （next_hidden_loss + this_output_loss）*
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        
        
        #  等到完成了所有反向传播误差计算， 才会更新权重矩阵，先暂时把更新矩阵存起来。
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)  # np.atleast_2d将输入的数组转化为至少两维：
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
    # 完成所有反向传播之后，更新权重矩阵。并把矩阵变量清零
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
   
    # print out progress
    if(j % 800 == 0):
        #print(synapse_0,synapse_h,synapse_1)
        print("总误差:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " - " + str(b_int) + " = " + str(out))
        print("------------")