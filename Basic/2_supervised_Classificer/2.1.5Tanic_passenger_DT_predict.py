#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
泰坦尼克号 乘客 存活与否预测  决策数 Decsion Tree  
'''

# 导入pandas用于数据分析
import pandas as pd
# 利用pandas的read_csv模块直接从互联网收集泰坦尼克号乘客数据
# titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# titanic.to_csv('raw_titanic.csv')
titanic = pd.read_csv('raw_titanic.csv')
# 观察一下前几行数据，可以发现，数据种类各异，数值型、类别型，甚至还有缺失数据
print titanic.head()
# 使用pandas，数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info()，查看数据的统计特性
titanic.info()

# 机器学习有一个不太被初学者重视，并且耗时，但是十分重要的一环，特征的选择，这个需要基于一些背景知识。根据我们对这场事故的了解，sex, age, pclass这些都很有可能是决定幸免与否的关键因素
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 对当前选择的特征进行探查
print X.info()

# 借由上面的输出，我们设计如下几个数据处理的任务：
# 1) age这个数据列，只有633个，需要补完
# 2) sex 与 pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替
# 首先我们补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)
# 对补完的数据重新探查
print X.info()
# 由此得知，age特征得到了补完

# 数据分割
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 33)


# 我们使用scikit-learn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print vec.feature_names_
# 同样需要对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))


# 从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
# 使用分割到的训练数据进行模型学习
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测
y_predict = dtc.predict(X_test)
# 输出预测准确性
print '决策树分类器准确度:',dtc.score(X_test, y_test)
# 从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report
# 输出更加详细的分类性能
print classification_report(y_predict, y_test, target_names = ['died', 'survived'])















