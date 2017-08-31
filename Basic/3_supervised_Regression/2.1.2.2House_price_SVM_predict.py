#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
美国波斯顿房价预测   SVM支持向量机 回归预测
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
###支持向量机模型########
# 从sklearn.svm中导入支持向量机（回归）模型
from sklearn.svm import SVR
# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel='linear')   # 模型
linear_svr.fit(X_train, y_train)    # 训练
linear_svr_y_predict = linear_svr.predict(X_test) # 预测

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr = SVR(kernel='poly')       # 模型
poly_svr.fit(X_train, y_train)      # 训练
poly_svr_y_predict = poly_svr.predict(X_test)     # 预测

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')         # 模型
rbf_svr.fit(X_train, y_train)       # 训练
rbf_svr_y_predict = rbf_svr.predict(X_test)       # 预测


###########结果评价###################
####线性核函数支持向量机
# 使用linear_svr模型自带的评估模块，并输出评估结果
print '线性核函数支持向量机LSVMR准确度： ', linear_svr.score(X_test, y_test)
# 使用R-squared、MSE和MAE指标对三种配置的支持向量机（回归）模型在相同测试集上进行性能评估
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 使用r2_score模块，并输出评估结果
print '确定系数 R-squared : ', r2_score(y_test, linear_svr_y_predict)
# 使用mean_squared_error模块，并输出评估结果
print '均值平方误差 Mean Squared Error: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print '均值绝对误差 Mean Absoluate Error: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))

######多项式核函数支持向量机####
# 使用poly_svr模型自带的评估模块，并输出评估结果
print '多项式核函数支持向量机poly_svr 模型准确度: ', poly_svr.score(X_test, y_test)
# 使用r2_score模块，并输出评估结果 s1=sum(真实值-预测值)^2  s2=sum(真实值-预测值均值)^2   R2 = 1-S1/S2
print '确定系数 R-squared : ', r2_score(y_test, poly_svr_y_predict)
# 使用mean_squared_error模块，并输出评估结果  sum(真实值-预测值均值)^2/m
print '均值平方误差 Mean Squared Error: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))
# 使用mean_absolute_error模块，并输出评估结果 sum|(真实值-预测值均值)|/m
print '均值绝对误差 Mean Absoluate Error: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))

######径向基核函数支持向量机####
# 使用rbf_svr模型自带的评估模块，并输出评估结果
print '径向基核函数支持向量机rbf_svr 模型准确度: ', rbf_svr.score(X_test, y_test)
# 使用r2_score模块，并输出评估结果 s1=sum(真实值-预测值)^2  s2=sum(真实值-预测值均值)^2   R2 = 1-S1/S2
print '确定系数 R-squared : ', r2_score(y_test, rbf_svr_y_predict)
# 使用mean_squared_error模块，并输出评估结果  sum(真实值-预测值均值)^2/m
print '均值平方误差 Mean Squared Error: ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
# 使用mean_absolute_error模块，并输出评估结果 sum|(真实值-预测值均值)|/m
print '均值绝对误差 Mean Absoluate Error: ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))












    







