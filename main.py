# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:09:20 2019

@author: 37112
"""

import time
import data
import utils
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
start = time.time()
#数据载入
data_path = "data/train-v2.0.json"
qlist, alist = data.read_corpus(data_path)
#展示词频最多的次
#word_dic = Counter([q for l in utils.cut(qlist) for q in l])
#utils.show_most_word_freq(word_dic, 50)
#载入预处理后问题
qlist_new = utils.load_qlist('data/q_prepro.txt')
#question = input("您想了解什么问题？")
question = "When did Beyonce start become popular"
# 使用tf-idf方法
idx = utils.find_top_similar_ask1(question, qlist_new)
alist = np.array(alist)
print(alist[idx])
end = time.time()
print(end - start, "s")