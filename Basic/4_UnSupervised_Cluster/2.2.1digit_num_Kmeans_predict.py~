#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
手写字体 聚类  K-means聚类算法  进行预测
'''
# 分别导入numpy、matplotlib以及pandas，用于数学运算、作图以及数据分析
import numpy as np                # 数学
import matplotlib.pyplot as plot  # 画图
import pandas as pd               # 数据处理

#######数据获取#######
# 使用pandas分别读取训练数据与测试数据集
## 从网络获取
#digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
#digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)
#digits_train.to_csv("MNIST/train.csv")
#digits_test.to_csv("MNIST/test.csv")
## 从本地打开(数据有点问题)
digits_train = pd.read_csv("MNIST/train.csv") # 28*28 + 1 =785 第一个为标签
digits_test  = pd.read_csv("MNIST/test.csv")

# 从训练与测试数据集上都分离出64维度的像素特征与1维度的数字目标
X_train = digits_train[np.arange(64)]
y_train = digits_train['label'] # 标签 
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]
# 从sklearn.cluster中导入KMeans模型
from sklearn.cluster import KMeans
# 初始化KMeans模型，并设置聚类中心数量为10
kmeans = KMeans(n_clusters=10)     # 模型
kmeans.fit(X_train)                # 训练
# 逐条判断每个测试图像所属的聚类中心
y_pred = kmeans.predict(X_test)    # 预测

#####结果评价#######
# 从sklearn导入度量函数库metrics
from sklearn import metrics
# 使用ARI进行KMeans聚类性能评估
print metrics.adjusted_rand_score(y_test, y_pred)








