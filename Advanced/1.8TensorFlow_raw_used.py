#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
tensorflow  
'''
import tensorflow as tf
import numpy as np
greeting = tf.constant('Hello Google Tensorflow!')

# 回话
sess = tf.Session()
result = sess.run(greeting)
print result
sess.close()



matrix1 = tf.constant([[3., 3.]]) #矩阵
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)# 矩阵乘法
linear = tf.add(product, tf.constant(2.0))#加法    A*X  + b
with tf.Session() as sess:
    result = sess.run(linear)
    print result


# 读取乳腺癌数据集
import pandas as pd
train = pd.read_csv('../Basic/1_breast-cancer/breast-cancer-train.csv')
test = pd.read_csv('../Basic/1_breast-cancer/breast-cancer-test.csv')

X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)

# 模型构建
b = tf.Variable(tf.zeros([1])) # 
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0)) # 权重1*2
y = tf.matmul(W, X_train) + b# 两唯线性模型

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_train)) #损失函数
optimizer = tf.train.GradientDescentOptimizer(0.01)#梯度下降优化算法 最小化损失函数
train = optimizer.minimize(loss)# 训练 优化器
# 初始化变量
init = tf.initialize_all_variables()
# 启动图 (graph)
sess = tf.Session()
sess.run(init)
# 拟合平面
for step in xrange(0, 1000): #进行1000次优化
    sess.run(train)#启动训练
    if step % 200 == 0:      #每200词打印一次优化结果  参数
        print step, sess.run(W), sess.run(b)


# 测试数据正负样例
 # 测试负例 厚度 大小
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
 # 测试正例
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# 画图显示测试数据正负样例
import matplotlib.pyplot as plot  # 画图 库

plot.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'], marker = 'o', s=200, c='red')  # 负例 红色
plot.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'], marker = 'x', s=150, c='black')# 正例 黑色
plot.xlabel(u'肿块厚度 Clump Thickness') #可能中文显示不正常 见 matplotlib显示中文
plot.ylabel(u'细胞大小 Cell Size')


lx = np.arange(0, 12)
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0]) / sess.run(W)[0][1]
plot.plot(lx, ly, color ='green')
plot.show()



