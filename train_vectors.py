# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lars, Lasso
import gc
import io
import re
import json
import scipy.spatial.distance as distance
import gensim.utils as utils
from six import iteritems
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

def valid_and_separate(line):
    vals = line.split(';')
    if len(vals) != 2:
        return None
    for i in range(2):
        v = re.sub(r'[\.\,\%«"\)\(]', '', vals[i]).strip()
        if not v:
            return None
        vals[i] = v
    if vals[0] == vals[1]:
        return None
    return vals


def readvocub(filename):
    tr = {}
    with io.open(filename, 'r', encoding='utf8') as fin:
        for line in fin:
            vals = valid_and_separate(line)
            if vals is None:
                continue
            tr[vals[0]] = [vals[1]]
    fin.close()
    return tr

folder = 'data'

source_corpora = folder + '\\eng_corpora.txt'
source_model_save = folder + '\\eng_model'

target_corpora = folder + '\\rus_corpora.txt'
target_model_save = folder + '\\rus_model'

final_file= folder +'\\bilingual.bin'

vector_size = 100
load_from_file = True

if load_from_file:
    target_model = Word2Vec.load(target_model_save)
    # target_model = Word2Vec.load_word2vec_format('E:\\NLp\\Data\\kaz_dataset_size(200).bin', binary=True)
else:
    print 'Train W2V for target'
    sentences = LineSentence(target_corpora)
    target_model = Word2Vec(sentences, size=vector_size, window=6, min_count=4, sg=1)
    target_model.save(target_model_save)

if load_from_file:
    source_model = Word2Vec.load(source_model_save)
    # source_model = Word2Vec.load_word2vec_format('E:\\NLp\\Data\\rus_dataset2_size(200).bin', binary=True)
else:
    print 'Train W2V for source'
    sentences = LineSentence(source_corpora)
    source_model = Word2Vec(sentences, size=vector_size, window=9, min_count=10, sg=1)
    source_model.save(source_model_save)

source2target = {}
target2source = {}
# load word pairs
source2target = readvocub('data\\eng_rus_vocab.txt')
# target2source = readvocub('E:\\Nlp\\Data\\kazakh_news_vocub_translations.txt')

sourcematrix=[]
targetmatrix=[]
cur = 0
for w_source in source_model.vocab:
    cur += 1
    if w_source in source2target:
        trans = source2target[w_source]
        for w_target in trans:
            if w_target in target_model.vocab:
                w_source_index = source_model.vocab[w_source].index
                w_target_index = target_model.vocab[w_target].index
                sourcematrix.append(source_model.syn0[w_source_index])
                targetmatrix.append(target_model.syn0[w_target_index])

source2target = None
target2source = None
gc.collect()
sourcematrix = np.array(sourcematrix)
targetmatrix = np.array(targetmatrix)
print('len of matricies', len(sourcematrix), len(targetmatrix))

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(200, input_dim=sourcematrix.shape[1], init='uniform'))
model.add(Activation('linear'))
# model.add(Dropout(0.5))
# model.add(Dense(200))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(targetmatrix.shape[1]))
model.add(Activation('linear'))
sgd = SGD(lr=0.05, decay=1e-6)
model.compile(loss='mean_squared_error',
              optimizer=sgd)

model.fit(sourcematrix, targetmatrix,
          nb_epoch=5,
          batch_size=64, verbose=1)
l = len(sourcematrix)
distances = np.zeros(l)
losses = np.zeros(l-1)
x_new = model.predict(sourcematrix,verbose=1)
avg = 0
for i in xrange(len(x_new)):
    dist = 1 - distance.cosine(targetmatrix[i],x_new[i])
    distances[i] = dist
for i in xrange(l -1):
    dist1 = distance.cosine(sourcematrix[i],sourcematrix[i+1])
    dist2 = distance.cosine(x_new[i],x_new[i+1])
    losses[i] = dist1 - dist2
print 'avg:', distances.mean()
print 'std:', distances.std()
print 'best:', distances.max()
print 'worst:', distances.min()
print 'loss:', losses.mean()