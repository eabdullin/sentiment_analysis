'''This example demonstrates the use of Convolution1D for text classification.
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py
Get to 0.835 test accuracy after 2 epochs. 100s/epoch on K520 GPU.
'''

from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten,Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import sklearn.cross_validation as valid
import sklearn.metrics as metrics
from keras.utils.visualize_util import plot

from IPython.display import SVG
from keras.utils.visualize_util import to_graph
# twit_parser = TweetDataParser()
# twit_parser.add_csv_file("E:\NLP\DATA\RUS\\twitter\\positive.csv",1)
# twit_parser.add_csv_file("E:\NLP\DATA\RUS\\twitter\\negative.csv",0)
# data = twit_parser.data
# print 'twitter processed'
# wordvectors = Word2VecBinReader()
# wordvectors.readvec('E:\\NLP\\DATA\\vectors\\rus_200_min_7.bin')
# y = np.array(twit_parser.sentiments)
# X, maxlen = wordvectors.make_idx_data(data, insert_new_words=False)
# max_features = len(wordvectors.vocub)
# print(max_features, len(X))
# print('word vectors processed')
# print('data splitted')


# set parameters:
batch_size = 324
word_vocub_len = 5000
word_maxlen= 100
char_vocub_len = 50
batch_size = 32
word_embedding_dims = 200
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

#build a model
word_cnn = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
word_cnn.add(Embedding(word_vocub_len, word_embedding_dims, input_length=word_maxlen))
word_cnn.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
word_cnn.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
word_cnn.add(MaxPooling1D(pool_length=2))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
word_cnn.add(Flatten())
word_cnn.add(Dense(250))
word_cnn.add(Dropout(0.5))


char_cnn = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
char_cnn.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1,
                           input_shape=(char_vocub_len,1)))
# we use standard max pooling (halving the output of the previous layer):
char_cnn.add(MaxPooling1D(pool_length=3))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
char_cnn.add(Flatten())
char_cnn.add(Dense(250))
char_cnn.add(Dropout(0.5))

model = Sequential()
model.add(Merge([word_cnn, char_cnn], mode='sum'))
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))
print('start compile')
model.compile(loss='binary_crossentropy',
              optimizer='adadelta')
print('model compilled')
# print('fit start')
# X_train, X_test, y_train, y_test = valid.train_test_split(X, y, test_size=0.3,)
# model.fit(X_train, y_train, batch_size=batch_size,
#           nb_epoch=nb_epoch, show_accuracy=True,
#           validation_data=(X_test, y_test))

# y_predscore = model.predict_proba(X_test, batch_size=10, verbose=1)
# print(y_predscore)
# vauc = metrics.roc_auc_score(y_test, y_predscore)
# y_pred = np.round(y_predscore)
# print(y_pred)
#
# acc = metrics.accuracy_score(y_test,y_pred)
# f_mes = metrics.f1_score(y_test,y_pred)
#
# print('validation accuracy = {:.3%}, f-measure = {:.3%},roc_auc ={:.3%}'.format( acc, f_mes,vauc)
plot(model, to_file='sentitweet_model_cnn.png')
# SVG(to_graph(model).create(prog='dot', format='svg'))