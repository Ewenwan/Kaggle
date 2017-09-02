#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
泰坦尼克号 数据 特征筛选 
'''

# 导入pandas用于数据分析
import pandas as pd
# 利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据
# titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# titanic.to_csv('raw_titanic.csv')
titanic = pd.read_csv('../Basic/2_supervised_Classificer/raw_titanic.csv')
# 观察一下前几行数据，可以发现，数据种类各异，数值型、类别型，甚至还有缺失数据
print titanic.head()
# 使用pandas，数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info()，查看数据的统计特性
titanic.info()
## 分离数据特征与预测目标
y_label   = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis = 1)
## 对对缺失数据进行填充
# 对年龄列特征值缺失值用该列均值代替
X['age'].fillna(X['age'].mean(), inplace=True)#使用平均数或者中位数都是对模型偏离造成最小影响的策略
# 其他列缺失值用UNKNOWN字符替换 其他列为字符型数据
X.fillna('UNKNOWN', inplace=True)# 其他 列缺失值NaN 用 UNKNOWN代替
## 分割数据，依然采样25%用于测试
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.25, random_state=33)

# 类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer() # 转化的数据必须为字典类型 其他类型需要先转化为字典类型
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 输出处理后特征向量的维度
print len(vec.feature_names_)

# 使用决策树模型依靠所有特征进行预测，并作性能评估
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy') # 信息熵为分裂属性
dt.fit(X_train, y_train)
print '未进行特征筛选的样本 决策树分类器准确度：'
print dt.score(X_test, y_test)




# 从sklearn导入特征筛选器
from sklearn import feature_selection
# 筛选前20%的特征，使用相同配置的决策树模型进行预测，并且评估性能
# 特征筛选器
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
# 进行训练数据的特征筛选
X_train_fs = fs.fit_transform(X_train, y_train)

dt_20 = DecisionTreeClassifier(criterion='entropy') # 信息熵为分裂属性
# 训练模型
dt_20.fit(X_train_fs, y_train)
# 进行测试数据的特征筛选
X_test_fs = fs.transform(X_test)
# 进行测试 
print '筛选前20%的特征的测试得分：'
print dt_20.score(X_test_fs, y_test)


# 通过交叉验证（下一节将详细介绍）的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化。
from sklearn.cross_validation import cross_val_score
import numpy as np
dt_k = DecisionTreeClassifier(criterion='entropy') # 信息熵为分裂属性
percentiles = range(1, 100, 2)  #特征筛选的百分比
results = []
for i in percentiles:
    # 不同比例的特征筛选器
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile = i)
    # 进行训练数据的特征筛选
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt_k, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print results
# 找到提现最佳性能的特征筛选的百分比
opt = np.where(results == results.max())[0][0]# 最优结果对应筛选比例数的位置
print opt
print '最优的特征筛选比例：'
print percentiles[opt]
'''
3
最优的特征筛选比例：
7
'''
#画图显示 各个特征筛选比例下的 准确度
import pylab as pl
pl.plot(percentiles, results)
pl.xlabel(u'特征筛选比例')
pl.ylabel(u'准确度')
pl.show()

# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
fs_opt = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
# 进行训练数据的特征筛选
X_train_fs_opt = fs_opt.fit_transform(X_train, y_train)
# 决策数模型
dt_opt = DecisionTreeClassifier(criterion='entropy') # 信息熵为分裂属性
# 训练
dt_opt.fit(X_train_fs_opt, y_train)
# 测试数据 特征筛选
X_test_fs_opt = fs_opt.transform(X_test)
print '最优的特征筛选下模型准确度：'
print dt_opt.score(X_test_fs_opt, y_test)


