import numpy as np
import sklearn.cross_validation as valid
import sklearn.metrics as metrics
from keras.constraints import unitnorm
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.visualize_util import plot
from tools.twit_data_parser import TweetDataParser
from tools.word2vecbinreader import Word2VecBinReader

twit_parser = TweetDataParser()
twit_parser.add_csv_file("E:\NLP\DATA\RUS\\twitter\\positive.csv",1)
twit_parser.add_csv_file("E:\NLP\DATA\RUS\\twitter\\negative.csv",0)
data = twit_parser.data
print('twitter processed')
wordvectors = Word2VecBinReader()
wordvectors.readvec('E:\\NLP\\DATA\\vectors\\rus_200_min_7.bin')
y = np.array(twit_parser.sentiments)
X, maxlen = wordvectors.make_idx_data(data, insert_new_words=False)
max_features = len(wordvectors.vocub)
print(max_features, len(X))
print('word vectors processed')

# set parameters:
batch_size = 324
word_vocub_len = 1000#wordvectors.W.shape[0]
word_maxlen= 100
char_vocub_len = 50
batch_size = 32
word_embedding_dims = 200#wordvectors.W.shape[1]
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2
print('data splitted')
batch_size = 324

model = Sequential()
model.add(Embedding(input_dim=word_vocub_len,output_dim=word_embedding_dims, input_length=word_maxlen))#, weights=[wordvectors.W], W_constraint=unitnorm()
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print('compile start')
model.compile(loss='binary_crossentropy',
              optimizer='adam')
print('model compilled')
# print('fit start')
# X_train, X_test, y_train, y_test = valid.train_test_split(X, y, test_size=0.3,)
# print(len(y_test))
# # for i in xrange(N_epoch):
# model.fit(X_train, y_train, batch_size=50, nb_epoch=1, verbose=1, show_accuracy=True)
# print('fitting ended')
#
# y_predscore = model.predict_proba(X_test, batch_size=10, verbose=1)
# print(y_predscore)
# vauc = metrics.roc_auc_score(y_test, y_predscore)
# y_pred = np.round(y_predscore)
# print(y_pred)
#
# acc = metrics.accuracy_score(y_test,y_pred)
# f_mes = metrics.f1_score(y_test,y_pred)
# print('validation accuracy = {:.3%}, f-measure = {:.3%},roc_auc ={:.3%}'.format( acc, f_mes,vauc)

# plot(model, to_file='sentitweet_model_rnn.png')


