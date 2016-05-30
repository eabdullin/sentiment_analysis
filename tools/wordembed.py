from sklearn.manifold import TSNE
import numpy as np
import numpy.core.multiarray as multiarray
import gensim.utils as utils
fromstring = multiarray.fromstring
dtype = multiarray.dtype
import codecs
import sys
import getopt
encoding='utf8'
unicode_errors='strict'
zeros = multiarray.zeros
def __init__():
    encoding='utf8'
def readvec(vecfile):
    with utils.smart_open(vecfile) as fin:
        header = utils.to_unicode(fin.readline(), encoding=encoding)
        vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
        vocab_size = int(1000)
        binary_len = np.dtype(np.float32).itemsize * vector_size
        weights_list = []
        xwords = []
        xweights = zeros((vocab_size, vector_size), dtype=np.float32)
        for line_no in range(vocab_size):
            # mixed text and binary: read text first, then binary
            word = []
            while True:
                ch = fin.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                    word.append(ch)
            word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
            weights = fromstring(fin.read(binary_len), dtype=np.float32)
            xwords.append(word)
            xweights[line_no] = weights
        return xwords, xweights

def finvec(word,xwords,xweights):
    for i in range(xwords):
        if xwords[i] == word:
            return xweights[i]
    return None
vec_file = 'E:\\NLP\\DATA\\vectors\\kaz_256_min_8.bin'
save_file = 'E:\\NLP\\DATA\\vectors\\kaz_2_min_8.txt'
try:
  opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
  print('wordembed.py -i <inputfile> -o <outputfile>')
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
     print('wordembed.py -i <inputfile> -o <outputfile>')
     sys.exit()
  elif opt in ("-i", "--ifile"):
     vec_file = arg
  elif opt in ("-o", "--ofile"):
     save_file = arg
print('Input file is "', vec_file)
print('Output file is "', save_file)
x_words, x_weights = readvec(vec_file)
print('\nstart tSNE processing')

perplexity = 40
model_tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=100, verbose=2)
np.set_printoptions(suppress=True)
x_data = model_tsne.fit_transform(x_weights)
print('\nstart saving')
with codecs.open(save_file,"w", "utf-8") as writetrain:
    for i in range(len(x_data)):
        writetrain.write('%s %f %f\n'%(x_words[i],x_data[i][0],x_data[i][1]))

