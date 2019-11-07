# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:08:51 2019

@author: 37112
"""

import json
import utils
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import string

def read_corpus(data_path):
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    让每一个问题和答案对应起来（下标位置一致）
    """    
    qlist = []
    alist = []
    with open(data_path) as f:
        data_all = json.load(f)['data']
        for data in data_all:
            paragraphs = data['paragraphs']
            for paragraph in paragraphs:
                qas = paragraph["qas"]
                for qa in qas:
                    if 'plausible_answers' in qa:
                        qlist.append(qa['question'])
                        alist.append(qa['plausible_answers'][0]['text'])
                    else:
                        qlist.append(qa['question'])
                        alist.append(qa['answers'][0]['text'])
    assert len(qlist) == len(alist)
    return qlist, alist

def preprocessing(qlist):
    """
    对于qlist做文本预处理操作。 可以考虑以下几种操作：
       停用词过滤、转换成lower_case、去掉一些无用的符号、去掉出现频率很低的词：比如出现次数少于10,20....
       对于数字的处理：分词完有些单词可能就是数字，把他们转成"#number"
       stemming(利用porter stemming)
    """
    qlist_data = utils.load_qlist('data/q_prepro.txt')
    stopset = set(stopwords.words('english'))
    minus_words = ["when", "what", "where", "how", "which", "who", "whom"]
    for i in minus_words:
        stopset.discard(i)
    p = PorterStemmer()
    qlist_ = utils.cut(qlist)
    word_dic = Counter([q for l in utils.cut(qlist_data) for q in l])
    low_freq_words = utils.find_low_freq_word(word_dic)
    new_list = []
#    f = open('data/q_prepro.txt', 'w', encoding='utf-8') 
    for line in qlist_:
        l = ""
        for word in line:
            word = word.lower()
            #stemming
            word = p.stem(word)
            # 去除所有标点符号
            word = ''.join(c for c in word if c not in string.punctuation)
            #数字转成”#number“
            if word.isdigit():
                word = "#number"
            #不是停用词且不是低频词，加入新list
            if word not in low_freq_words and word not in stopset:
                l += word + " "
#        f.write(l +'\n')
        new_list.append(l)
#    f.close()
    return new_list
   
    
if __name__ == "__main__":
    data_path = "data/train-v2.0.json"
    qlist, alist = read_corpus(data_path)
#    print(qlist, alist)
#    preprocessing(qlist)
    