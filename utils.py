# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:07:47 2019

@author: 37112
"""

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


#分词，因为是英文，所以可以直接split，中文用jieba
def cut(input_list):
    #句子是问题，用split分词最后一个单词就包含“？”，需要用“”代替。
    list_new = []
    for i in input_list:
        list_new.append(i.replace("?", "").split())
    return list_new

def show_most_word_freq(word_dict, topN):
    #选取词典中最多的N的次用以展示
    most_ = word_dict.most_common(topN)
    x, y = [], []
    for i in range(len(most_)):
        x.append(most_[i][0])
        y.append(most_[i][1])
    plt.plot(x, y)
    plt.xlabel("words")
    plt.ylabel("frequence")
    plt.show()
 
def find_low_freq_word(word_dict):
    low_freq_word = []
    for key, value in word_dict.items():
        if value <= 5:
            low_freq_word.append(key)
    return low_freq_word

def find_top_similar_ask1(q, qlist):
    #使用tf-idf的方法
    q = data.preprocessing([q])
    vectorizer = TfidfVectorizer()
    #方法1：使输入与全部问题库的信息作比较
#    x = vectorizer.fit_transform(qlist)
#    input_vec = vectorizer.transform(q)
    #方法2：从倒排表中取出相关联的索引，仅与有相同字的问题作比较
    index_list = []
    invert_table = load_inverse_table()
    for c in cut(q):
        for i in c:
            if i in invert_table.keys():
                values = invert_table[i]
                for value in values:
                    index_list.append(int(value))
#                index_list += invert_table[i]
    index_list = list(set(index_list))
    qlist = np.asarray(qlist)
    x = qlist[index_list]
    x = vectorizer.fit_transform(x)
    input_vec = vectorizer.transform(q)
    res = cosine_similarity(input_vec, x)
    n = np.argmax(res)
    return n

def get_glove_data():
    with open('data/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        emb = []
        vocab = []
        for line in lines:
            row = line.strip().split()
            vocab.append(row[0])
            emb.append(row[1:])
    f.close()
    return emb, vocab

def get_words_vec(q, emb, vocab):
    # 获取句子的vec值
    glove_index_list = []
    # 取句子对应单词的索引
    for word in q.split():
        if word in vocab:
            index = vocab.index(word)
            glove_index_list.append(index)
    return emb[glove_index_list].astype(float).sum()/len(q.split())

def find_top_similar_ask2(q, qlist):
    #使用glove的方法
    #输入问题q，返回答案
    # 1 将问题和问题库里的问题进行embedding，转换成句子向量
    # 2 从倒排表里筛选候选
    # 3 计算问题与候选问题的相似度
    # 4 选取相似度取值最大的
    q = data.preprocessing([q])
    q = "".join(q)
    emb, vocab = get_glove_data()
    emb = np.asarray(emb)
    # 输入问题的vec值
    q_vec = get_words_vec(q, emb, vocab)
    qlist_vec = []
    i = 0
    for qlist_ in qlist:
        qlist_vec.append(get_words_vec(qlist_, emb, vocab))
        i += 1
        if i % 5000 == 0:
            print(i)
    qlist_vec = np.asarray(qlist_vec)
    abs_v = abs(qlist_vec[0])
    # 存储所有返回结果的索引
    res = []
    # 从倒排表中取出相关联的索引
    index_list = []
    invert_table = load_inverse_table()
    for c in cut([q]):
        for i in c:
            if i in invert_table.keys():
                values = invert_table[i]
                for value in values:
                    index_list.append(int(value))
#                index_list += invert_table[i]
    index_list = list(set(index_list))
    # 遍历倒排表内所有值，将list中所有vec值与input_q的vec值做对比，绝对值差较小的数的索引存入res中
#    for value in qlist_vec[index_list]:
#        if abs(q_vec - value) < abs_v:
#            abs_v = abs(q_vec - value)
#            res.append(qlist_vec[index_list].tolist().index(value))
    values = qlist_vec[index_list]
    res = cosine_similarity(q_vec, values)
    n = np.argmax(res)

    return n
    
#倒排表
def inverse_table(qlist, word_dict):
    table = {}
    for key, value in word_dict.items():
        table[key] = []
    for i, q in enumerate(cut(qlist)):
        for word in q:
            if word in table.keys():
                table[word].append(i) 
    f = open('data/table.txt', 'w', encoding='utf-8') 
    for k, v in table.items():
        f.write(str(k) + ' ' + str(v) + '\n')
    f.close()

#载入倒排表
def load_inverse_table():
    dict_temp = {}
    # 打开文本文件
    f = open('data/table.txt','r', encoding='utf-8')
    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in f.readlines():
        line = line.strip()
        data = line.split()
        k = data[0]
        rep = ['[', ',', ']']
        for i in range(1, len(data)):
            for j in data[i]:
                if j in rep:
                    id = data[i].index(j)
                    data[i] = data[i][:id] + data[i][id+1:]
        v = list(data[1:])
        dict_temp[k] = v
    f.close()
    return dict_temp
    
#载入预处理后问题
def load_qlist(qlist_path):
    qlist = []
    f = open(qlist_path,'r', encoding='utf-8')
    for line in f.readlines():
        line = line.strip()
        qlist.append(line)
    f.close()
    return qlist

if __name__ == "__main__":
    data_path = "data/train-v2.0.json"
    qlist, alist = data.read_corpus(data_path)
    qlist_new = load_qlist('data/q_prepro.txt')
    word_dic = Counter([q for l in cut(qlist_new) for q in l])
    table = load_inverse_table('data/table.txt')
    print(table)

    