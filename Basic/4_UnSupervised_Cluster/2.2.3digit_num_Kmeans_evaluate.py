#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
手写字体 聚类  K-means聚类算法   轮廓系数评价聚类结果
'''
import numpy as np # 数学计算
from sklearn.cluster import KMeans # 聚类算法 
from scipy.spatial.distance import cdist # 
import matplotlib.pyplot as plt          # 画图

# 使用均匀分布函数随机三个簇，每个簇周围10个数据样本
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

# 绘制30个数据样本的分布图像
X = np.hstack((cluster1, cluster2, cluster3)).T
plt.scatter(X[:,0], X[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
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
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()

















