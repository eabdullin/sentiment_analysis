import json
import numpy as np
import codecs
from sklearn.svm import SVR
import gensim.utils as utils
from gensim.models import Word2Vec
import statsmodels.api as sm
from .word2vecbinreader import Word2VecBinReader
from sklearn.linear_model import Ridge
from . import gls_correction_matrix
import scipy.spatial.distance as distance
import gc
import scipy.optimize as opt
from sklearn.decomposition import PCA
from scipy.linalg import toeplitz
from . import yandex_translate

ruskaz = {}
kazrus = {}
# with codecs.open('E:\\NlpData\\en_ru_kaz-dictionary.json','r',encoding='utf8') as file:
#     data = json.load(file)
#     for t in data["Translations"]:
#         indexrus = str(t["Rus"])
#         indexkaz = str(t["Kaz"])
#         rusword = data["Words"][indexrus]["Text"].lower();
#         kazword = data["Words"][indexkaz]["Text"].lower();
#         if rusword not in ruskaz:
#             ruskaz[rusword] = []
#         elif kazword not in ruskaz[rusword]:
#             ruskaz[rusword].append(kazword)
#         if kazword not in kazrus:
#             kazrus[kazword] = []
#         elif rusword not in kazrus[kazword]:
#             kazrus[kazword].append(rusword)
w = Word2Vec()

w.load_word2vec_format()
w.save_word2vec_format()
ruskaz = yandex_translate.readvocub('E:\\NlpData\\rus_kfu_news_vocub_translations.txt')
kazrus = yandex_translate.readvocub('E:\\NlpData\\kazakh_news_vocub_translations.txt')
print(len(ruskaz))
# print len(kazrus)
rusvecreader = Word2VecBinReader()
rusvecreader.readvec('E:\\NlpData\\rus_dataset2_size(200).bin')
kazvecreader = Word2VecBinReader()
kazvecreader.readvec('E:\\NlpData\\kaz_dataset_size(200).bin')
print('rus vectors', len(rusvecreader.word_idx_map))
print('kaz vectors', len(kazvecreader.word_idx_map))
rusmatrix = []
kazmatrix = []
cur = 0
size = len(rusvecreader.word_idx_map)
for wru in rusvecreader.word_idx_map:
    # if cur % 10000 == 0:
    #     print '\r%d from %d'%(cur,size)
    cur += 1
    if wru in ruskaz:
        trans = ruskaz[wru]
        added = False
        for wkaz in trans:
            if added:
                break
            if wkaz in kazvecreader.word_idx_map:
                added = True
                wruindex = rusvecreader.word_idx_map[wru]
                wkazindex = kazvecreader.word_idx_map[wkaz]
                rusmatrix.append(rusvecreader.W[wruindex])
                kazmatrix.append(kazvecreader.W[wkazindex])
    # elif wru in kazvecreader.word_idx_map:
    #     wruindex = rusvecreader.word_idx_map[wru]
    #     wkazindex = kazvecreader.word_idx_map[wru]
    #     rusmatrix.append(rusvecreader.W[wruindex])
    #     kazmatrix.append(kazvecreader.W[wkazindex])
ruskaz = None
kazrus = None
gc.collect()
# print 'correlation', \
#     np.corrcoef(rusmatrix,kazmatrix)
rusmatrix = np.array(rusmatrix)
kazmatrix = np.array(kazmatrix)
print('len of matricies', len(rusmatrix), len(kazmatrix))

# m = []
# for i in xrange(kazmatrix.shape[1]):
#     print i
#     clf = SVR(C=1.0, epsilon=0.2)
#     clf.fit(rusmatrix, kazmatrix[:,i])
#     m.append(clf.coef0)
#
# A = np.array(m)
# m = None
r = Ridge()
r.predict()
X.dot( r.coef_.T,
                dense_output=True) + self.intercept_
A, resids, rank, st = np.linalg.lstsq(rusmatrix, kazmatrix)
# print A.shape
# A = gls_correction_matrix.GLSAR(rusmatrix,kazmatrix)
# opt.leastsq()
d,l = gls_correction_matrix.test(A,rusmatrix,kazmatrix)
# newr = np.array(rusmatrix[d > 0.4])
# newk = np.array(kazmatrix[d > 0.4])
# print len(newr)
# A, resids, rank, st = np.linalg.lstsq(newr,newk)
# d,l = gls_correction_matrix.test(A,newr,newk)
w = Word2Vec()
final_rus_matrix = np.dot(rusvecreader.W,A)
with utils.smart_open('rus_kaz_200.bin', 'wb')as fout:
    fout.write(utils.to_utf8("%s %s\n" % (rusvecreader.W.shape[0]+kazvecreader.W.shape[0],200)))
    for i in range(len(rusvecreader.vocub)):
        word = 'rus__'+rusvecreader.vocub[i]
        row = final_rus_matrix[i]
        fout.write(utils.to_utf8(word) + b" " + row.tostring())
    print('end rus')
    for i in range(len(kazvecreader.vocub)):
        word = 'kaz__'+kazvecreader.vocub[i]
        row = kazvecreader.W[i]
        fout.write(utils.to_utf8(word) + b" " + row.tostring())
    print('end kaz')

