#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
手写字体 特征降维  主成份分析  PCA 
'''
import pandas as pd               # 数据处理
import numpy as np
#######数据获取#######
# 使用pandas分别读取训练数据与测试数据集
## 从网络获取
'''
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)
digits_train.to_csv("MNIST/train.csv")
digits_test.to_csv("MNIST/test.csv")
'''
## 从本地打开(数据有点问题)
digits_train = pd.read_csv("MNIST/train.csv")
digits_test  = pd.read_csv("MNIST/test.csv")

# 分割训练数据的特征向量和标记
X_digits = digits_train[map(str,np.arange(64))]# 安照 关键字方式过滤
y_digits = digits_train['64']

# 从sklearn.decomposition导入PCA
from sklearn.decomposition import PCA
# 初始化一个可以将高维度特征向量（64维）压缩至2个维度的PCA
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)

# 显示10类手写体数字图片经PCA压缩后的2维空间分布
from matplotlib import pyplot as plt

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in xrange(len(colors)):
	px = X_pca[:, 0][y_digits.as_matrix() == i] # 各种数字  提取出来
	py = X_pca[:, 1][y_digits.as_matrix()== i]
	plt.scatter(px, py, c=colors[i]) #对于各种数字 不同颜色显示
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel(u'第一主成份')
    plt.ylabel(u'第二主成份')
    plt.show()
# 显示 降维后的训练数据
plot_pca_scatter()


# 对训练数据、测试数据进行特征向量（图片像素）与分类目标的分隔
X_train = digits_train[map(str,np.arange(64))]
y_train = digits_train['64'] # 标签 
X_test = digits_test[map(str,np.arange(64))]
y_test = digits_test['64']

# 导入基于线性核的支持向量机分类器
from sklearn.svm import LinearSVC
# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并在测试数据上做出预测，存储在y_predict中
svc = LinearSVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)

# 使用PCA将原64维的图像数据压缩到20个维度
estimator = PCA(n_components=20)
# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化（transform）
pca_X_test = estimator.transform(X_test)
# 使用默认配置初始化LinearSVC，对压缩过后的20维特征的训练数据进行建模，并在测试数据上做出预测，存储在pca_y_predict中

pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)

# 从sklearn.metrics导入classification_report用于更加细致的分类性能分析
from sklearn.metrics import classification_report
# 对使用原始图像高维像素特征训练的支持向量机分类器的性能作出评估
print svc.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names=np.arange(10).astype(str))
# 对使用PCA压缩重建的低维图像特征训练的支持向量机分类器的性能作出评估
print pca_svc.score(pca_X_test, y_test)
print classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str))











