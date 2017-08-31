#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
新闻数据文本分类  NB  朴素贝叶斯分类器 
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# 从sklearn.datasets里导入新闻数据抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
# 与之前预存的数据不同，fetch_20newsgroups需要即时从互联网下载数据
# news = fetch_20newsgroups(subset='all')

# new_train = fetch_20newsgroups(subset='train')
# new_test  = fetch_20newsgroups(subset='test')
# 或者直接载入下载好的新闻数据   http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
# 解压后将两个文件夹内数据放在一起
# news_train = load_files('20news-bydate/20news-bydate-train')
# news_test = load_files('20news-bydate/20news-bydate-test')
# 查验数据规模和细节
# print len(news_train.data)
# print len(news_test.data)
# print news_train.data[0]
# print type(news_train)

# 加载全部数据
news = load_files('20news-bydate/all_news')
# 查验数据规模和细节
print len(news.data)
print news.data[0]

# 从sklearn.cross_validation 导入 train_test_split 分割数据
from sklearn.cross_validation import train_test_split
# 随机采样25%的数据样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块。
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train) # 提取字典  形成字典单词 使用记录向量
X_test = vec.transform(X_test)


# 从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 从使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 对测试样本进行类别预测，结果存储在变量y_predict中
y_predict = mnb.predict(X_test)

#模型自身准确度报告
print '朴素贝叶斯分类器准确度： ', mnb.score(X_test, y_test)
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names = news.target_names)








