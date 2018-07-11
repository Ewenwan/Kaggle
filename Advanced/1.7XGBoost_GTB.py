#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
集成训练模型   
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
X = titanic[['pclass', 'age', 'sex']]
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


# 使用随机森林分类器进行集成模型的训练以及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()  # 模型
rfc.fit(X_train, y_train)       # 训练
rfc_y_pred = rfc.predict(X_test)# 预测
# 输出预测准确性
print '随机森林分类器准确度:',rfc.score(X_test, y_test)
# 从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report
# 输出更加详细的分类性能 精确率、召回率、F1指标
print classification_report(rfc_y_pred, y_test, target_names = ['died', 'survived'])



# 导入 XGBoot集成学习模型
from xgboost import XGBClassifier
xgbc = XGBClassifier() #模型
xgbc.fit(X_train, y_train)#训练
# 打印准确度

print '测试数据集上 eXtreme Gradient Boosting Classifier 模型准确度： ', xgbc.score(X_test, y_test)
