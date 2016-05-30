import numpy as np
import numpy.core.multiarray as ma
encoding = 'utf8'
unicode_errors = 'strict'
import scipy.spatial.distance as dist

totalcount = 0
alignedcount = 0
from math import *


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


def finvec(word, xwords, xweights):
    for i in range(len(xwords)):
        if xwords[i] == word:
            return xweights[i]
    return None




vru_words = None
vkk_words = None
vkk_vecs = None
vru_vecs = None
reader = Word2VecBinReader()
reader.readvec('E:/NlpData/rus_mixed_size(200)window(7)negative(1)cbow(0).bin')
# vru_words, vru_vecs = readvec('E:/NlpData/rus_mixed_size(200)window(7)negative(1)cbow(0).bin')
# vkk_words, vkk_vecs = readvec('E:/NlpData/kaz_mixed_size(200)window(7)negative(1)cbow(0).bin')
print('len1', len(vru_words))
print('len2', len(vkk_words))
matrix_rukk = np.zeros((len(vru_words), len(vkk_words)), dtype=np.float32)
i = 0
j = 0

for word in vru_words:
    wordvec = finvec(word, vkk_words, vkk_vecs)
    totalcount += 1
    if wordvec is not None:
        alignedcount += 1
        for wordi in xrange(len(vkk_words)):
            vec = vkk_vecs[wordi]
            # dotres = np.linalg.norm(wordvec - vec)
            dotres = cosine_similarity(wordvec,vec)
            matrix_rukk[i, j] = dotres
            j += 1
    j = 0
    i += 1
i = 0
j = 0

for word in vkk_words:
    wordvec = finvec(word,vru_words,vru_vecs)
    if wordvec is not None:
        for wordi in range(len(vru_words)):
            vec = vru_vecs[wordi]
            dotres = cosine_similarity(wordvec,vec)
            matrix_rukk[j,i] = (matrix_rukk[j,i] + dotres)/2
            j += 1
    j = 0
    i += 1
print('total: {}, aligned; {}'.format(totalcount, alignedcount))
print('saving')
with open('rukkmatrix.txt', 'w') as w:
    for i in range(matrix_rukk.shape[1]):
        for j in range(matrix_rukk.shape[0]):
            w.write('{0}\t'.format(matrix_rukk[i, j]))
        w.write('\n')
