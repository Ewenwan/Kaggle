#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
词向量模型  
'''
# 从sklearn.datasets中导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups
# 导入numpy，并且重命名为np
import numpy as np

# 与之前预存的数据不同，fetch_20newsgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')

X, y = news.data, news.target

from bs4 import BeautifulSoup

import nltk, re

# 定义一个函数  用于 剥离新闻中的句子 成一个句子列表
def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:\n",
	sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences

# 剥离新闻文本中的句子 用于训练
sentences = []
for x in X:
    sentences += news_to_sentences(x)

#导入词向量模型
from gensim.models import word2vec
#配置词向量维度
num_features = 300
# 保证被考虑的词汇维度
min_word_count = 20 
# CPU 核心数
num_workers = 2
# 定义训练词向量的上下文窗口大小
context = 5 
# 下采样   频率
downsampling = 1e-3

from gensim.models import word2vec
# 训练词向量模型
model = word2vec.Word2Vec(sentences, workers=num_workers, \
	size=num_features, min_count = min_word_count, \
	window = context, sample = downsampling)
# 设定当前训练好的词向量为最终版，可以加快模型训练速度
model.init_sims(replace=True)

# 利用模型，寻找训练文本中于email最相关的10个词汇
model.most_similar('morning') # 


