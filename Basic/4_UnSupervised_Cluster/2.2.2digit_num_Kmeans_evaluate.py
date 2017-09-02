#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
 K-means聚类算法   轮廓系数评价聚类结果
'''
# 导入numpy
import numpy as np
# 从sklearn.cluster中导入KMeans算法包
from sklearn.cluster import KMeans
# 从sklearn.metrics导入silhouette_score用于计算轮廓系数
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 分割出3*2=6个子图，并在1号子图作图
plt.subplot(3,2,1)   #三行两列图
# 初始化原始数据点
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(zip(x1, x2)).reshape(len(x1), 2)#点对

# 在1号子图做出原始数据点阵的分布
plt.xlim([0, 10])  #坐标轴范围
plt.ylim([0, 10])
plt.title(u'实例') #图题
plt.scatter(x1, x2)#散点图  图1
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']  #颜色
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+'] #形状
clusters = [2, 3, 4, 5, 8] #聚类中心个数
subplot_counter = 1#图下标 变量
sc_scores = []
for t in clusters:
    subplot_counter += 1 #图下标 变量增加1
    plt.subplot(3, 2, subplot_counter)#画图2
    kmeans_model = KMeans(n_clusters=t).fit(X)#聚类
    for i, l in enumerate(kmeans_model.labels_):# 点下标   中心下边
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')# 聚类好的点
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')# 评价
    sc_scores.append(sc_score)
    # 绘制轮廓系数与不同类簇数量的直观显示图
    plt.title(u'K = %s, 轮廓系数= %0.03f' %(t, sc_score))#各种聚类中心数量 的聚类结果得分
# 绘制轮廓系数与不同类簇数量的关系曲线
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel(u'聚类中心数量')
plt.ylabel(u'轮廓系数Silhouette Coefficient Score')

plt.show()
    











