# -*- coding: utf-8 -*-
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import codecs
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import random
import codecs
from word2vecbinreader import Word2VecBinReader


def train2d_transformation(vec_file, method='PCA', save_file=None, n=None, rand=True, verbose=1):
    transformator = None
    if method == 'PCA':
        transformator = PCA(2)
    elif method == 'tSNE':
        transformator = TSNE(n_components=2, perplexity=40, verbose=verbose)
    else:
        raise ValueError('method is not find')
    if verbose > 0:
        print('read word vectors from', vec_file)
    reader = Word2VecBinReader()
    reader.readvec(vec_file)
    if verbose > 0:
        print('read word vectors ended')
        print('start %s processing' % method)

    if n is None:
        n = len(reader.W)
    if rand:
        r = range(len(reader.W))
        r = random.sample(r, n)
    else:
        r = range(n)
    x_data = transformator.fit_transform(reader.W[r])
    if verbose > 0:
        print('end of %s processing' % method)
    if save_file is not None:
        if verbose > 0:
            print('start saving to', save_file)
        with codecs.open(save_file, "w", "utf-8") as writetrain:
            for i in range(len(x_data)):
                writetrain.write('%s;%f,%f\n' % (reader.vocub[r[i]], x_data[i][0], x_data[i][1]))
        if verbose > 0:
            print('save ended')
    words = []
    for i in r:
        words.append(reader.vocub[i])
    return words, x_data


def save_as_picture(x_words, X_data, filename, xinches=80, yinches=80, dpi=200):
    xmax, ymax = X_data.max(axis=0)
    xmin, ymin = X_data.min(axis=0)
    matplotlib.rc('font', **{'sans-serif': 'Arial', 'family': 'sans-serif'})
    fig = plt.figure(frameon=False)
    fig.set_size_inches(xinches, yinches)
    fig.suptitle('kaz rus words', fontsize=50, fontweight='bold')
    ax = fig.add_subplot(111, autoscale_on=True, xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_axis_off()
    for i in range(len(X_data)):
        color = 'black'
        word = x_words[i]
        if x_words[i].startswith('rus__'):
            color = 'green'
            word = x_words[i].replace('rus__', '')
        elif x_words[i].startswith('kaz__'):
            color = 'blue'
            word = x_words[i].replace('kaz__', '')
        ax.text(X_data[i][0], X_data[i][1], word, color=color, fontsize=10)
    fig.savefig(filename, dpi=dpi)


def open_wordmap(file):
    x_words = []
    with codecs.open(file, 'r', encoding='utf8') as file:
        x_data = []
        for line in file:
            vals = line.split(';')
            if len(vals):
                continue
            coord = vals[1].split(',')
            if len(coord) != 2:
                continue
            x = np.float32(coord[0])
            y = np.float32(coord[1])
            x_data.append([x, y])
            x_words.append(vals[0])
        X_data = np.array(x_data)
        return x_words, X_data
    return None


# data = range(200)
# r = range(len(data))
# r = random.sample(r, 20)
# print data[r]

x_words, X_data = train2d_transformation('E:\\NLP\\nlppython\\ml_course\\final_project\\heroes.bin', method='tSNE', rand=False)

save_as_picture(x_words, X_data, 'E:\\NLP\\nlppython\\ml_course\\final_project\\heroes.png',xinches=20,yinches=20)
# x_words,X_data = train2d_tsne('E:\NlpData\kaz_rus_2_vec_200_cbow.bin',n_sne=7000)
# save_as_picture(x_words,X_data,'kaz_rus_2_vec_200_cbow.png')
