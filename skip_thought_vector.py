# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: skip_thought_vector
Description : 
Author : arlen
date：18-6-27
------------------------------------------------- """
import sys
import numpy as np
import pickle
from keras import Model, callbacks
from keras.layers import GRU, Bidirectional, Input, Dense, RepeatVector
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed
import utils

max_sent_length = 30
emd_dim = 300

inputs = Input((max_sent_length, emd_dim))
encoded = Bidirectional(GRU(512, activation='relu', return_sequences=False))(inputs)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(512, activation='tanh')(encoded)
encoder = Model(inputs, encoded)

decoded_1 = Bidirectional(GRU(512, activation='relu', return_sequences=True))(RepeatVector(30)(encoded))
decoded_1 = TimeDistributed(Dense(512, activation='relu'))(decoded_1)
decoded_1 = TimeDistributed(Dense(512, activation='relu'))(decoded_1)
decoded_1 = TimeDistributed(Dense(300, activation='linear'))(decoded_1)

decoded_2 = Bidirectional(GRU(512, activation='relu', return_sequences=True))(RepeatVector(30)(encoded))
decoded_2 = TimeDistributed(Dense(512, activation='relu'))(decoded_2)
decoded_2 = TimeDistributed(Dense(512, activation='relu'))(decoded_2)
decoded_2 = TimeDistributed(Dense(300, activation='linear'))(decoded_2)

skipthought = Model(inputs, [decoded_1, decoded_2])
skipthought.compile(optimizer=Adam(lr=0.001), loss='mean_squared_logarithmic_error')


def train():
    callbackslist = callbacks.ModelCheckpoint(filepath='./model/best_model.hdf5',
                                              monitor='val_loss',
                                              mode='auto',
                                              period=1,
                                              save_best_only=True,
                                              save_weights_only=True)

    with open('./data/tripe.pkl', 'rb') as f:
        tripe = pickle.load(f)
        pres, mids, nexs = tripe

    skipthought.fit(mids, [pres, nexs],
                    epochs=5,
                    validation_split=0.02,
                    batch_size=512,
                    callbacks=[callbackslist])


def calculate_similarity(v1, v2):
    fraction = np.dot(v1, v2)
    denominator = (np.linalg.norm(v1) * (np.linalg.norm(v2))) + 0.0001
    return fraction / denominator


if __name__ == '__main__':

    if sys.argv[1] == 'tarin':
        train()

    elif sys.argv[1] == 'test':
        w2v = utils.W2V()
        skipthought.load_weights('./model/best_model.hdf5')

        q1 = '市民如何辨别假冒电动车和如何维权呢'
        q2 = '记者教大家怎样分辨假的电动车并且维护你的权益'
        q3 = '记者在采访中了解到，南宁速派奇专卖店出售的电动车售价一般比同行低'
        q4 = '７月２日，记者在向业内人士请教后，归纳成以下四点：（一）看价格，如价格明显低于同行就要警惕了'

        v1 = encoder.predict([[w2v.sent2vec(q1)]])[0]
        v2 = encoder.predict([[w2v.sent2vec(q2)]])[0]
        v3 = encoder.predict([[w2v.sent2vec(q3)]])[0]
        v4 = encoder.predict([[w2v.sent2vec(q4)]])[0]

        print('q1 and q2: %0.2f' % calculate_similarity(v1, v2))
        print('q1 and q2: %0.2f' % calculate_similarity(v1, v3))
        print('q1 and q2: %0.2f' % calculate_similarity(v1, v4))