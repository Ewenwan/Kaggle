#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
 Skflow   将TensorFLow封装为 Sklearn 接口形式
'''

#######数据获取#######################
########################################
# 从sklearn.datasets导入波士顿房价数据读取器
'''
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中
boston = load_boston()
# 输出数据描述
print boston.DESCR
'''
from sklearn import datasets, metrics, preprocessing, cross_validation
# 从读取房价数据存储在变量boston中
boston = datasets.load_boston()
#X, y = boston.data, boston.target
X = boston.data       # 数据集
y = boston.target     # 标签
# 随机采样25%的数据构建测试样本，其余作为训练样本
# 从sklearn.cross_validation导入数据分割器
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

# 分别初始化对特征和目标值的标准化器
scaler = preprocessing.StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
# 注意只接受二维数组 
# 一维数组 array.reshape(-1, 1) 变成单列的二维数组   单个特征 
# array.reshape(1, -1)          变成单行的二维数组   单个样本
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.transform(y_test.reshape(-1,1))

import skflow
tf_lr = skflow.TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
tf_lr.fit(X_train, y_train)
tf_lr_y_predict = tf_lr.predict(X_test)
#### 
print 'TensorFLow线性回归模型：'
print '均值绝对误差：', metrics.mean_absolute_error(tf_lr_y_predict, y_test)

print '均值平方误差：', metrics.mean_squared_error(tf_lr_y_predict, y_test)

print '准确度：', metrics.r2_score(tf_lr_y_predict, y_test)



# 深度神经网络模型 隐含层 数量 [100, 40]
tf_dnn_regressor = skflow.TensorFlowDNNRegressor(hidden_units=[100, 40],\
                                          steps=10000, learning_rate=0.01, batch_size=50)
# 训练   
tf_dnn_regressor.fit(X_train, y_train)
#预测
tf_dnn_regressor_y_predict = tf_dnn_regressor.predict(X_test)
print 'TensorFLow深度神经网络模型：'
print '均值绝对误差：', metrics.mean_absolute_error(tf_dnn_regressor_y_predict, y_test)

print '均值平方误差：', metrics.mean_squared_error(tf_dnn_regressor_y_predict, y_test)

print '准确度：', metrics.r2_score(ttf_dnn_regressor_y_predict, y_test)




# 从sklearn.ensemble中导入RandomForestRegressor、ExtraTreesGressor以及GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
# 使用RandomForestRegressor训练模型，并对测试数据做出预测，结果存储在变量rfr_y_predict中
rfr = RandomForestRegressor() # 随机森林回归模型
rfr.fit(X_train, y_train)     # 训练
rfr_y_predict = rfr.predict(X_test)# 预测
print 'sklearn随机森林模型：'
print '均值绝对误差：', metrics.mean_absolute_error(tf_dnn_regressor_y_predict, y_test)

print '均值平方误差：', metrics.mean_squared_error(tf_dnn_regressor_y_predict, y_test)

print '准确度：', metrics.r2_score(ttf_dnn_regressor_y_predict, y_test)
