#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
聚类  K-means聚类算法   肘部观察法选择最佳中心数量
'''
import numpy as np # 数学计算
from sklearn.cluster import KMeans # 聚类算法 
from scipy.spatial.distance import cdist # 
import matplotlib.pyplot as plt          # 画图

# 使用均匀分布函数随机三个簇，每个簇周围10个数据样本
cluster1 = np.random.uniform(0.5, 1.5, (2, 10)) #(最小值，最大值，大小(行，列))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

# 绘制30个数据样本的分布图像
X = np.hstack((cluster1, cluster2, cluster3)).T # 绑成一个 列表  列表
plt.scatter(X[:,0], X[:, 1]) # 散点图
plt.xlabel('x1')#横坐标 
plt.ylabel('x2')#纵坐标
plt.show()


# 测试9种不同聚类中心数量下，每种情况的聚类质量，并作图
K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel(u'平均离散度 Average Dispersion')
plt.title(u'肘部观察法选择最佳中心数量')
plt.show()

















