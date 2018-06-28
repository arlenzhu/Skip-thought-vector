# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: utils
Description : 
Author : arlen
date：18-6-26
------------------------------------------------- """
import pandas as pd
import re
import numpy as np
import jieba
import pickle


def load_data():
    df = pd.read_csv('./data/news_cnt.txt', encoding='utf-8', names=['cnt'])['cnt'].tolist()
    corpus = []
    for line in df:
        sent = [i for i in re.split('[。？！.?!”"；;]', line) if len(i) > 2]
        corpus.append(sent)

    return corpus[:10000]


def load_word2vec():
    """
    读取预训练word2vec词向量
    :return:
    """
    with open('w2v/sgns.sogou.bigram', 'r', encoding='utf-8') as f:
        f.readline()
        word_vec = dict()
        for line in f:
            line_ = line.split(' ')
            key_ = line_[0]
            val_ = np.array(list(map(float, line_[1: -1])))
            word_vec[key_] = val_
    return word_vec


class W2V(object):
    def __init__(self):
        self.w2v = load_word2vec()

    def sent2vec(self, sent):
        sent_vec = np.zeros((30, 300), np.float)
        sent = jieba.lcut(sent)[:30]
        for i, k in enumerate(sent):
            sent_vec[i, :] = self.w2v.get(k, np.zeros(300))
        return sent_vec


def make_tripe(corpus, w2v):
    pres, mids, nexs = list(), list(), list()
    for doc in corpus:
        for idx in range(1, len(doc) - 2, 1):
            pres.append(w2v.sent2vec(doc[idx - 1]))
            mids.append(w2v.sent2vec(doc[idx]))
            nexs.append(w2v.sent2vec(doc[idx + 1]))
    return np.array(pres), np.array(mids), np.array(nexs)


if __name__ == '__main__':
    corpus = load_data()
    w2v = W2V()
    (pres, mids, nexs) = make_tripe(corpus, w2v)
    open('./data/tripe.pkl', 'wb').write(pickle.dumps((pres, mids, nexs)))
