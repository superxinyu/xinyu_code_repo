关键词：numpy、rnn、nn、bp
用numpy模拟RNN对两个二进制数字进行相减
numpy模拟RNN中输入层、隐层、输出层的构造，并通过正向传播计算损失，通过反向传播计算梯度并更新权重
数据就是两个数字的二进制相减，每一位的两个数字作为输入，输出为该位置对应的相减结果
因为每一位的结果不仅依赖于当前位的输出，还严重依赖于前一位是否借位，所以用RNN模拟