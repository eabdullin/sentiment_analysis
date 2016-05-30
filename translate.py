# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from sklearn.linear_model import Ridge
import gc
import io
import re
import scipy.spatial.distance as distance

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

folder = 'E:\\NLP\\DATA\\corpuses'

source_corpora = folder + '\\rus_corpora.txt'
source_model_save = folder + '\\rus_model'

target_corpora = folder + '\\kaz_corpora.txt'
target_model_save = folder + '\\kaz_model'

final_file='sourse_target_200.bin'

vector_size = 100
load_from_file = True

if load_from_file:
    #     sourse_model = Word2Vec.load(source_model_save)
    sourse_model = Word2Vec.load_word2vec_format('E:\\NLp\\Data\\rus_dataset2_size(200).bin', binary=True)
else:
    print 'Train W2V for sourse'
    sentences = LineSentence(source_corpora)
    source_model = Word2Vec(sentences, size=vector_size, window=9, min_count=10, sg=1)
    source_model.save(source_model_save)

if load_from_file:
    #     target_model = Word2Vec.load(target_model_save)
    target_model = Word2Vec.load_word2vec_format('E:\\NLp\\Data\\kaz_dataset_size(200).bin', binary=True)
else:
    print 'Train W2V for target'
    sentences = LineSentence(target_corpora)
    target_model = Word2Vec(sentences, size=vector_size, window=6, min_count=4, sg=1)
    target_model.save(target_model_save)

sourse2target = {}
target2sourse = {}
#load word pairs
sourse2target = readvocub('E:\\Nlp\\Data\\rus_kfu_news_vocub_translations.txt')
target2sourse = readvocub('E:\\Nlp\\Data\\kazakh_news_vocub_translations.txt')

soursematrix=[]
targetmatrix=[]
cur = 0
for w_source in sourse_model.vocab:
    cur += 1
    if w_source in sourse2target:
        trans = sourse2target[w_source]
        added = False
        for w_target in trans:
            if w_target in target_model.vocab:
                w_source_index = sourse_model.vocab[w_source].index
                w_target_index = target_model.vocab[w_target].index
                soursematrix.append(sourse_model.syn0[w_source_index])
                targetmatrix.append(target_model.syn0[w_target_index])

sourse2target = None
target2sourse = None
gc.collect()
soursematrix = np.array(soursematrix)
targetmatrix = np.array(targetmatrix)
print('len of matricies', len(soursematrix), len(targetmatrix))

r = Ridge(alpha=0.1)
r.fit(soursematrix, targetmatrix)

x_new = r.predict(soursematrix)
avg = 0
for i in xrange(len(x_new)):
    dist = 1 - distance.cosine(targetmatrix[i],x_new[i])
    avg += dist
print(avg/len(x_new))
