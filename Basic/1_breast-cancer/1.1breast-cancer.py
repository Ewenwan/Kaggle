#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
良性/恶性 乳腺癌 肿瘤数据显示 分类测试
'''
# 测试代码
# python  1.1breast-cancer.py
import pandas as pd #数据处理库

df_train = pd.read_csv('breast-cancer-train.csv')
df_test  = pd.read_csv('breast-cancer-test.csv')
# 测试数据正负样例
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']] # 测试负例 厚度 大小
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']] # 测试正例
# 画图显示测试数据正负样例
import matplotlib.pyplot as plot  # 画图 库

plot.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')  # 负例 红色
plot.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')# 正例 黑色
plot.xlabel(u'肿块厚度 Clump Thickness') #可能中文显示不正常 见 matplotlib显示中文
plot.ylabel(u'细胞大小 Cell Size')
plot.show()



# 加入 随机线性方程 直线
import numpy as np
intercept = np.random.random([1]) # 产生 一个 0~1之间的随机数  
coef = np.random.random([2])      # 产生 两个 0~1之间的随机数  系数
lx = np.arange(0, 12)             # 产生 [0 1 2 3 ... 11]列表
ly = (-intercept - lx * coef[0]) / coef[1]# 随机线性方程
plot.plot(lx, ly, c='yellow')             # 画图
# 显示上面的散点图
plot.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plot.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plot.xlabel(u'肿块厚度 Clump Thickness')
plot.ylabel(u'细胞大小 Cell Size')
plot.show()


# 逻辑回归 分类测试
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()# 逻辑回归模型
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])# 前10个数据用于训练
print 'Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])#测试结果显示   0.868571428571
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
plot.plot(lx, ly, c='green')
plot.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plot.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plot.xlabel(u'肿块厚度 Clump Thickness')
plot.ylabel(u'细胞大小 Cell Size')
plot.show()



# 全部训练数据用于训练
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type']) #全部 524个样本数据用于训练
print 'Testing accuracy (all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']) # 显示测试结果  0.937142857143
intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept - lx * coef[0]) / coef[1]
plot.plot(lx, ly, c='blue')
plot.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'], marker = 'o', s=200, c='red')
plot.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'], marker = 'x', s=150, c='black')
plot.xlabel(u'肿块厚度 Clump Thickness')
plot.ylabel(u'细胞大小 Cell Size')
plot.show()






















