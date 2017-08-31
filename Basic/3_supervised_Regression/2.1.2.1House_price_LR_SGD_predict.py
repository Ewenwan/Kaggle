#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
美国波斯顿房价预测    LR 线性回归   SGD随机梯度下降
'''

#######数据获取#######
# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中
boston = load_boston()
# 输出数据描述
print boston.DESCR

#########数据分割#########
# 从sklearn.cross_validation导入数据分割器
from sklearn.cross_validation import train_test_split
# 导入numpy并重命名为np
X = boston.data       # 数据集
y = boston.target     # 标签
# 随机采样25%的数据构建测试样本，其余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

#####数据分析######
# 分析回归目标值的差异
import numpy as np
print "目标值最大值：", np.max(boston.target)  # 最大值
print "目标值最小值：", np.min(boston.target)  # 最小值
print "目标值均值  ：", np.mean(boston.target) # 均值

#####数据处理######
# 从sklearn.preprocessing导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
# 分别对训练和测试数据的特征以及目标值进行标准化处理
# 注意只接受二维数组 
# 一维数组 array.reshape(-1, 1) 变成单列的二维数组   单个特征 
# array.reshape(1, -1)          变成单行的二维数组   单个样本
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))

######模型训练及预测#################
###线性回归模型########
# 从sklearn.linear_model导入LinearRegression
from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归器LinearRegression
lr = LinearRegression()    # 线性回归模型
# 使用训练数据进行参数估计
lr.fit(X_train, y_train)   # 训练
# 对测试数据进行回归预测
lr_y_predict = lr.predict(X_test)# 预测
###随机梯度下降SGD回归模型#####
# 从sklearn.linear_model导入SGDRegressor  随机梯度下降（SGD）回归
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()                # 随机梯度下降（SGD）回归模型
# 使用训练数据进行参数估计
sgdr.fit(X_train, y_train)           # 训练
sgdr_y_predict = sgdr.predict(X_test)# 预测


######结果评价######
###LR模型结果####
# 使用LinearRegression模型自带的评估模块，并输出评估结果
print '线性回归模型LR准确度： ', lr.score(X_test, y_test)
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 使用r2_score模块，并输出评估结果
print '确定系数 R-squared : ', r2_score(y_test, lr_y_predict)
# 使用mean_squared_error模块，并输出评估结果
print '均值平方误差 Mean Squared Error: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print '均值绝对误差 Mean Absoluate Error: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))
######SGD模型结果####
# 使用SGDRegressor模型自带的评估模块，并输出评估结果
print '随机梯度下降SGD模型准确度: ', sgdr.score(X_test, y_test)
# 使用r2_score模块，并输出评估结果 s1=sum(真实值-预测值)^2  s2=sum(真实值-预测值均值)^2   R2 = 1-S1/S2
print '确定系数 R-squared : ', r2_score(y_test, sgdr_y_predict)
# 使用mean_squared_error模块，并输出评估结果  sum(真实值-预测值均值)^2/m
print '均值平方误差 Mean Squared Error: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))
# 使用mean_absolute_error模块，并输出评估结果 sum|(真实值-预测值均值)|/m
print '均值绝对误差 Mean Absoluate Error: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))











    







