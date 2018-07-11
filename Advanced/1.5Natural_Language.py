#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
自然语言处理  NLTH
'''

sent1 = 'The cat is walking in the bedroom.'     # 语言样本1
sent2 = 'A dog was running across the kitchen.'  # 语言样本2
from sklearn.feature_extraction.text import CountVectorizer # 词次数统计  量化模型
count_vec = CountVectorizer() #模型 
sentences = [sent1, sent2]    #样本
print count_vec.fit_transform(sentences).toarray()#得到词量化结果    统计成字典后 记录各个单词的使用次数
'''
[[0 1 1 0 1 1 0 0 2 1 0]
[1 0 0 1 0 0 1 1 1 0 1]]
'''
print count_vec.get_feature_names()# 字典各维度 特征名称
# [u'across', u'bedroom', u'cat', u'dog', u'in', u'is', u'kitchen', u'running', u'the', u'walking', u'was']

# 导入TfidfVectorizer文本抽取器   # 词权重 统计
from sklearn.feature_extraction.text import TfidfVectorizer
# 模型
tfidf_vec = TfidfVectorizer()
print tfidf_vec.fit_transform(sentences)
print tfidf_vec.get_feature_names()

# 导入 自然语言处理  NLTH模块
import nltk
tokens_1 = nltk.word_tokenize(sent1)
print tokens_1 #输出词表
tokens_2 = nltk.word_tokenize(sent2)
print tokens_2
vocab_1 = sorted(set(tokens_1))  # 按ASCII码排序
vocab_2 = sorted(set(tokens_2))

# 原始词根
stemmer = nltk.stem.PorterStemmer()
tem_1 = [stemmer.stem(t) for t in tokens_1]
stem_2 = [stemmer.stem(t) for t in tokens_2]

# 词性标注器
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
print pos_tag_1
pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print pos_tag_2
