#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
超参数 搜索  网络搜索 GridSearch 和 并行搜索 Parallel GridSearch
'''
####################################
#网络搜索 GridSearch
####################################
# 从sklearn.datasets中导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups
# 导入numpy，并且重命名为np
import numpy as np

# 与之前预存的数据不同，fetch_20newsgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')
# new_train = fetch_20newsgroups(subset='train')
# new_test  = fetch_20newsgroups(subset='test')
# 或者直接载入下载好的新闻数据   http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
# 解压后将两个文件夹内数据放在一起
#from sklearn.datasets import load_files
# news_train = load_files('20news-bydate/20news-bydate-train')
# news_test = load_files('20news-bydate/20news-bydate-test')
# 查验数据规模和细节
# print len(news_train.data)
# print len(news_test.data)
# print news_train.data[0]
# print type(news_train)

# 加载全部数据   有编码问题
#news = load_files('../Basic/2_supervised_Classificer/20news-bydate/all_news')
# 从sklearn.cross_validation 导入 train_test_split 分割数据
from sklearn.cross_validation import train_test_split
# 随机采样25%的数据样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 导入支持向量机（分类）模型
from sklearn.svm import SVC
# 从sklearn.feature_extraction.text里分别导入TfidfVectorizer
# 导入TfidfVectorizer文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer
#导入Pipeline
from sklearn.pipeline import Pipeline
#使用Pipeline 简化系统搭建流程，将文本抽取与分类器模型串联起来。
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
# 这里需要试验的2个超参数的的个数分别是4、3， svc__gamma的参数共有10^-2, 10^-1... 。
# 这样我们一共有12种的超参数组合，12个不同参数下的模型。
parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}
# 从sklearn.grid_search中导入网格搜索模块GridSearchCV。
from sklearn.grid_search import GridSearchCV
# 将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知GridSearchCV。
# 请大家务必注意refit=True这样一个设定
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行单线程网格搜索。
time = gs.fit(X_train, y_train)
print time
print gs.best_params_, gs.best_score_

# 输出最佳模型在测试集上的准确性
print gs.score(X_test, y_test)





#####################################
# 并行搜索 Parallel GridSearch
#######################################

# 从sklearn.datasets中导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups
# 导入numpy，并且重命名为np
import numpy as np

# 与之前预存的数据不同，fetch_20newsgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')
# new_train = fetch_20newsgroups(subset='train')
# new_test  = fetch_20newsgroups(subset='test')
# 或者直接载入下载好的新闻数据   http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
# 解压后将两个文件夹内数据放在一起
#from sklearn.datasets import load_files
# news_train = load_files('20news-bydate/20news-bydate-train')
# news_test = load_files('20news-bydate/20news-bydate-test')
# 查验数据规模和细节
# print len(news_train.data)
# print len(news_test.data)
# print news_train.data[0]
# print type(news_train)

# 加载全部数据   有编码问题
#news = load_files('../Basic/2_supervised_Classificer/20news-bydate/all_news')
# 从sklearn.cross_validation 导入 train_test_split 分割数据
from sklearn.cross_validation import train_test_split
# 随机采样25%的数据样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 导入支持向量机（分类）模型
from sklearn.svm import SVC
# 从sklearn.feature_extraction.text里分别导入TfidfVectorizer
# 导入TfidfVectorizer文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer
#导入Pipeline
from sklearn.pipeline import Pipeline
#使用Pipeline 简化系统搭建流程，将文本抽取与分类器模型串联起来。
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
# 这里需要试验的2个超参数的的个数分别是4、3， svc__gamma的参数共有10^-2, 10^-1... 。
# 这样我们一共有12种的超参数组合，12个不同参数下的模型。
parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}
# 从sklearn.grid_search中导入网格搜索模块GridSearchCV。
from sklearn.grid_search import GridSearchCV
# 将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知GridSearchCV。
# 请大家务必注意refit=True这样一个设定
# 初始化配置并行网格搜索，n_jobs=-1代表使用该计算机全部的CPU。
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)

# 执行多线程并行网格搜索。
time = gs.fit(X_train, y_train)
print time
gs.best_params_, gs.best_score_
# 输出最佳模型在测试集上的准确性
print gs.score(X_test, y_test)
