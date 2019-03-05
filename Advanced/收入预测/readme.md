# 收入预测

#### 根据人口普查的数据来预测个人的年收入是否超过50K

比赛链接：[kaggle复赛收入预测](https://www.kaggle.com/c/smlc)

### 个人年收入预测

个人年收入预测是指通过对人口普查信息的分析和研究，探索个人收入情况的变化规律，并以此规律去完成预测的过程。现有一组人口普查的数据，并借此进行个人年收入预测，进而更好的了解民生情况。

## 赛制介绍

本次比赛系机器学习方向2016级内部竞赛的第二次竞赛（复赛），旨在通过相对简单的应用场景建模分析，完整地完成比赛过程。通过本次比赛，学生应该掌握数据集预处理的基本方法以及建模分析预测数据的方法等。

### 赛程安排

**时间**：10月10日-10月30日。

**流程**：

-1. 每支参赛队伍注册一个kaggle账号，并登入，然后打开指定链接，可以查看相应的竞赛要求和数据集。
-2. 从该平台下载数据集，本地调试算法，并在该平台上提交结果，提交完成后，即得相应的分数并可以查看公共排名结果。（注意：公共排名仅供参考，比赛结束后，最终排名以私有排名为准）
-3. 每支参赛队伍每天有且仅有二次提交机会，高分成绩直接覆盖低分成绩。
-4. 比赛提供标准训练集（含标签），仅供参赛选手训练算法模型；提供标准测试数据集（无标签），供参赛选手提交评测结果参与排名。比赛中严禁使用外部数据集，一经发现，成绩作废。
-5. 比赛中需要使用代码管理工具对代码进行管理，并在提交结果之后存储当前代码作为提交成绩依据的记录，比赛结束公布最终排名时需提交取得最高成绩的代码。
-6. 比赛截止时，将对所有参赛队伍代码进行审核，识别并剔除只靠人工标注而没有算法贡献的队伍与使用重复代码参赛的队伍。

### 参赛对象

机器学习方向全体学生，并以小组为单位参赛。

### 成绩评定

根据比赛所设评价标准对参赛队伍提交的结果进行成绩评定并排名。比赛完成后将所有队伍分成三组：

**优秀组**：排名靠前6组。奖励包括较大幅度的平时成绩奖励、期末成绩奖励、其它奖励。

**合格组**：奖励包括平时成绩奖励、期末成绩奖励。

**不屈组**：排名靠后6组。无奖励。

### 比赛组织

软件学院机器学习教研室

## 赛题与数据

### 介绍

根据人口普查的数据来预测个人的年收入是否超过50K

### 数据集

数据集包含以下两个文件：

| 文件名称         | 文件格式       |
| ---------------- | --------------|
| `train-data.txt` | .txt（3.79M）  |
| `testx.csv`      | .csv（755 KB） |

`train-data.txt`为训练数据集，其中包含了众多用户人口普查的相关信息及对应的年收入情况（即标签）。`testx.csv`为测试数据集（样本不含对应的年收入情况），当模型训练完成之后，需要对测试集进行预测得到预测结果

数据集中的各个属性介绍如下：

| 序号 | 属性名           | 属性取值范围                                                 | 说明                 |
| ---- | ---------------- | ------------------------------------------------------------ | ------------------ |
| 1    | `age`            | Z[1]                                                         | 年龄。              |
| 2    | `workclass`      | {Private,Self-emp-not-inc,Self-emp-inc,Federal-gov,Local-gov,State-gov,Without-pay,Never-worked,`?`[2]} | 分别代表工作的类型。 |
| 3    | `fnlwgt`         | Z                                                            | 最终的权重系数[3]。  |
| 4    | `education`      | {Bachelors,Some-college,11th,HS-grad,Prof-school,Assoc-acdm,Assoc-voc,9th,7th-8th,12th,Masters,1st-4th,10th,Doctorate,5th-6th,Preschool} | 教育背景。           |
| 5    | `education-num`  | Z                                                            | 教育时间。           |
| 6    | `marital-status` | {Married-civ-spouse,Divorced,Never-married,Separated,Widowed,Married-spouse-absent,Married-AF-spouse} | 婚姻状况。           |
| 7    | `occupation`     | {Tech-support,Craft-repair,Other-service,Sales,Exec-managerial,Prof-specialty,Handlers-cleaners,Machine-op-inspct,Adm-clerical,Farming-fishing,Transport-moving,Priv-house-serv,Protective-serv,Armed-Forces,`?`} | 职业情况。           |
| 8    | `relationship`   | {Wife,Own-child,Husband,Not-in-family,Other-relative,Unmarried} | 亲朋关系情况。    |
| 9    | `race`           | {White,Asian-Pac-Islander,Amer-Indian-Eskimo,Other,Black}    | 种族情况。           |
| 10   | `sex`            | {Female,Male}                                                | 性别情况。           |
| 11   | `capital-gain`   | Z                                                            | 资本收益情况。       |
| 12   | `capital-loss`   | Z                                                            | 资本损失情况。       |
| 13   | `hours-per-week` | Z                                                            | 每周工作的小时数。   |
| 14   | `native-country` | 详情[4]                                                      | 故乡情况。           |

数据集标签说明：每一行文本代表一个样本标记，其中`<=50K`表示负类样本，即年收入未超过50K；`>50K`表示正类样本，与负类样本的意义相反。

### 提交说明

预测测试集的结果的提交格式参考`sample.csv`中的格式进行提交。要求必须提交`.csv`格式文件，且每条结果与测试集的数据集一一对应。

**规定**：预测结果中正类样本用1表示，负类样本用0表示。

------

**脚注**：

[1] Z 表示正整数

[2] `?`在数据集中为缺失值，需要自行处理。

[3] 具有相似人口统计特征的人应具有相似的权重

[4] 故乡情况包括以下几种：United-States,Cambodia,England,Puerto-Rico,Canada,Germany,Outlying-US(Guam-USVI-etc),India,Japan,Greece,South,China,Cuba,Iran,Honduras,Philippines,Italy,Poland,Jamaica, Vietnam,Mexico,Portugal,Ireland,France,Dominican-Republic,Laos,Ecuador,Taiwan,Haiti,Columbia,Hungary, Guatemala,Nicaragua,Scotland,Thailand,Yugoslavia,El-Salvador,Trinadad&Tobago,Peru,Hong,Holand-Netherlands,`?`。

### 评价指标

The evaluation metric for this competition is [Mean F1-Score](https://www.kaggle.com/wiki/MeanFScore). The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision p and recall r. Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the ratio of true positives to all actual positives (tp + fn). The F1 score is given by:

![](CodeCogsEqn.png)

The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.


# 代码

```python
# coding: utf-8


#导入必要的工具包和数据集
import pandas as pd
import numpy as np
train_data=pd.read_csv('F:/PyWork/match2/trainx.csv',header=None)
test_data=pd.read_csv('F:/PyWork/match2/testx.csv')



#使训练集测试集格式一致：给训练集加上特征名，分离出标签列
train_y=train_data[14]
train_data=train_data.drop(columns=[14])
feature_names=test_data.columns
train_data.columns=feature_names




#改变capital-gain/loss的范围，进行对数转换
skewed = ['capital-gain', 'capital-loss']
train_set = pd.DataFrame(data = train_data)
test_set = pd.DataFrame(data = test_data)
train_set[skewed] = train_data[skewed].apply(lambda x: np.log(x + 1))
test_set[skewed] = test_data[skewed].apply(lambda x: np.log(x + 1))




#正则化数值特征，将其规范到0-1区间
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

train_set[numerical] = scaler.fit_transform(train_set[numerical])

test_set[numerical] = scaler.fit_transform(test_set[numerical])




#删除不重要的特征
train=train_set.drop(columns=['fnlwgt','education','native-country'])
test=test_set.drop(columns=['fnlwgt','education','native-country'])


#填补缺失值
train['workclass']=train['workclass'].replace(' ?',' Private')
train['occupation']=train['occupation'].replace(' ?',' Prof-specialty')
# train['native-country']=train['native-country'].replace(' ?',' United-States')

test['workclass']=test['workclass'].replace(' ?',' Private')
test['occupation']=test['occupation'].replace(' ?',' Prof-specialty')
# test['native-country']=test['native-country'].replace(' ?',' United-States')


#对标签列进行编码
income_set = set(train_y)
#print(income_set)
train_y= train_y.map({' <=50K': 0, ' >50K': 1}).astype(int)
# train_y


#编码类别特征
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in ['workclass','marital-status','occupation','relationship','race','sex']:
    train[i]=le.fit_transform(train[i])

for i in ['workclass','marital-status','occupation','relationship','race','sex']:
    test[i]=le.fit_transform(test[i])



#进行独热编码
train = pd.get_dummies(train)
test=pd.get_dummies(test)



#数据集划分
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,train_y,test_size=0.6,random_state=33)



#进行模型的学习，并用格点搜索选择最优参数组合
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)



#使用格点搜索调参
#cv_params={'n_estimators':[375,400,425]}
#other_params = {
#    'n_estimators': 400,
#    'booster': 'gbtree',
#    'objective': 'binary:logistic',
#    'max_depth': 5,
#    'subsample': 1.0,
#    'colsample_bytree': 0.3,
#    'min_child_weight': 1,
#    'learning_rate': 0.1,
#    'gamma':0.3
#} 
#model=xgb.XGBClassifier(**other_params)
#optimized_GBM=GridSearchCV(estimator=model,param_grid=cv_params,scoring='f1',cv=5,verbose=1)
#optimized_GBM.fit(X_train,y_train)
#evalute_result=optimized_GBM.grid_scores_
#print('每轮迭代运行结果:{0}'.format(evalute_result))
#print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))



#使用最优参数组合进行模型的学习以及评价
model=xgb.XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=5, min_child_weight=1, 
                        subsample=1.0, colsample_bytree=0.3, gamma=0.3)
model.fit(X_train, y_train)

ans = model.predict(X_test)
from sklearn.metrics import f1_score
ans = model.predict(X_test)
print(f1_score(y_test, ans))



#使用模型预测并导出到csv
pred = model.predict(test)
print(pred)
data1 = pd.DataFrame(pred)
data1.to_csv('data11.csv')

```

