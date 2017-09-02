#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
特征提升   新闻数据 词计数统计 词频统计  
在有无去除 停用词的情况下 朴素贝叶斯分类器
'''
# 定义一组字典列表，用来表示多个数据样本（每个字典代表一个数据样本）
measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.}, {'city': 'San Fransisco', 'temperature': 18.}]
# 从sklearn.feature_extraction 导入 DictVectorizer (字典数据向量化)
from sklearn.feature_extraction import DictVectorizer
# 初始化DictVectorizer特征抽取器
vec = DictVectorizer()
# 输出转化之后的特征矩阵 字符型成独立列特征
print vec.fit_transform(measurements).toarray()
# 输出各个维度的特征含义
print vec.get_feature_names()
'''
[[  1.   0.   0.  33.]
 [  0.   1.   0.  12.]
 [  0.   0.   1.  18.]]
['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']
'''

# 从sklearn.datasets里导入新闻数据抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
# 与之前预存的数据不同，fetch_20newsgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')

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

# 加载全部数据   有编码问题
#news = load_files('../Basic/2_supervised_Classificer/20news-bydate/all_news')
# 从sklearn.cross_validation 导入 train_test_split 分割数据
from sklearn.cross_validation import train_test_split
# 随机采样25%的数据样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块。
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
# 只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_mnb_train = vec.fit_transform(X_train) # 提取字典  形成字典单词 使用记录向量
X_mnb_test = vec.transform(X_test)


# 从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 从使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 使用朴素贝叶斯分类器，对CountVectorizer（不去除停用词）后的训练样本进行参数学习
mnb.fit(X_mnb_train, y_train)
# 使用模型进行预测
y_predict = mnb.predict(X_mnb_test)
#模型自身准确度报告
print '计数统计样本 朴素贝叶斯分类器准确度： ', mnb.score(X_mnb_test, y_test)
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names = news.target_names)


# 从sklearn.feature_extraction.text里分别导入TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# 采用默认的配置对TfidfVectorizer进行初始化（默认配置不去除英文停用词），并且赋值给变量tfidf_vec
tfidf_vec = TfidfVectorizer() # 词权重  Term frequency * Inverse Doc Frequency
# 使用tfidf的方式，将原始训练和测试文本转化为特征向量
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)
# 依然使用默认配置的朴素贝叶斯分类器，在相同的训练和测试数据上，对新的特征量化方式进行性能评估

# 从使用默认配置初始化朴素贝叶斯模型
mnb_tfidf = MultinomialNB()
# 使用朴素贝叶斯分类器，对CountVectorizer（不去除停用词）后的训练样本进行参数学习
mnb_tfidf.fit(X_tfidf_train, y_train)
# 使用模型进行预测
y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
#模型自身准确度报告
print '词频词权重统计样本 朴素贝叶斯分类器准确度： ', mnb_tfidf.score(X_tfidf_test, y_test)
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print classification_report(y_test, y_tfidf_predict, target_names = news.target_names)




#分别使用停用词过滤配置初始化CountVectorizer与TfidfVectorizer
count_filter_vec= CountVectorizer(analyzer='word', stop_words='english')
tfidf_filter_vec= TfidfVectorizer(analyzer='word', stop_words='english')
# 使用带有停用词过滤的CountVectorizer对训练和测试文本分别进行量化处理
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)
# 初始化默认配置的朴素贝叶斯分类器，并对CountVectorizer后的数据进行预测与准确性评估
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)
#模型自身准确度报告
print '计数统计样本 过滤停用次 朴素贝叶斯分类器准确度： ', mnb_count_filter.score(X_count_filter_test, y_test)
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print classification_report(y_test, y_count_filter_predict, target_names = news.target_names)

# 使用带有停用词过滤的TfidfVectorizer对训练和测试文本分别进行量化处理
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)
# 初始化另一个默认配置的朴素贝叶斯分类器，并对TfidfVectorizer后的数据进行预测与准确性评估
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)
#模型自身准确度报告
print '词频词权重统计样本 过滤停用次 朴素贝叶斯分类器准确度： ', mnb_count_filter.score(X_tfidf_filter_test, y_test)
# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print classification_report(y_test, y_tfidf_filter_predict, target_names = news.target_names)















