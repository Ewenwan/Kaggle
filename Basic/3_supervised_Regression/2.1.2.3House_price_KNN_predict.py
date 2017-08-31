#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
美国波斯顿房价预测   KNN  回归预测  
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
# 从sklearn.neighbors导入KNeighborRegressor（K近邻回归器）
from sklearn.neighbors import KNeighborsRegressor
# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归：weights='uniform'
uni_knr = KNeighborsRegressor(weights='uniform')  # 均匀 权重 模型
uni_knr.fit(X_train, y_train)                     # 训练
uni_knr_y_predict = uni_knr.predict(X_test)       # 预测

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归：weights='distance'
dis_knr = KNeighborsRegressor(weights='distance') # 距离加权回归模型
dis_knr.fit(X_train, y_train)                     # 训练
dis_knr_y_predict = dis_knr.predict(X_test)       # 预测

###########结果评价###################
####均匀 权重 模型 KNN
# 使用uni_knr模型自带的评估模块，并输出评估结果
print '均匀权重最近邻分类器uniform-KNN准确度： ', uni_knr.score(X_test, y_test)
# 使用R-squared、MSE和MAE指标对三种配置的支持向量机（回归）模型在相同测试集上进行性能评估
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 使用r2_score模块，并输出评估结果
print '确定系数 R-squared : ', r2_score(y_test, uni_knr_y_predict)
# 使用mean_squared_error模块，并输出评估结果
print '均值平方误差 Mean Squared Error: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print '均值绝对误差 Mean Absoluate Error: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))

####距离加权回归模型 KNN
# 使用dis_knr模型自带的评估模块，并输出评估结果
print '距离加权最近邻分类器distance-KNN准确度： ', dis_knr.score(X_test, y_test)
# 使用R-squared、MSE和MAE指标对三种配置的支持向量机（回归）模型在相同测试集上进行性能评估
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 使用r2_score模块，并输出评估结果
print '确定系数 R-squared : ', r2_score(y_test, dis_knr_y_predict)
# 使用mean_squared_error模块，并输出评估结果
print '均值平方误差 Mean Squared Error: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print '均值绝对误差 Mean Absoluate Error: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))

