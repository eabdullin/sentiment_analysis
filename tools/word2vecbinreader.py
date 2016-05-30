import numpy as np
import numpy.core.multiarray as multiarray
import io
import gensim
class Word2VecBinReader:

    def __init__(self):
        self.last_word_index = 0
        self.dim = 0
        self.W = None
        self.word_idx_map = {}
        self.vocub = []

    def readvec(self, vecfile):
        """
        read pre-trained word vectors from binary file
        :param vecfile: file path
        :return: self
        """
        with io.open(vecfile) as fin:
            header = fin.readline()
            vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
            binary_len = np.dtype(np.float32).itemsize * vector_size
            # vocab_size = 5000
            self.W = np.zeros((vocab_size, vector_size), dtype=np.float32)
            for index in xrange(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = unicode(b''.join(word), encoding='utf-8')
                weights = multiarray.fromstring(fin.read(binary_len), dtype=np.float32)
                self.vocub.append(word)
                self.word_idx_map[word] = index
                self.W[index] = weights
                self.last_word_index = index
        return self

    def get_idx_from_sent(self, sent, insert_new_words=False):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        words = sent.split()
        for word in words:
            if word in self.word_idx_map:
                x.append(self.word_idx_map[word])
            elif insert_new_words:
                self.word_idx_map[word] = self.last_word_index
                x.append(self.last_word_index)
                self.vocub.append(self.last_word_index)
                self.last_word_index += 1
        return x

    def make_idx_data(self, text_sentences, insert_new_words=False, max_l=None):
        """
        Transforms sentences into a 2-d matrix.
        """
        data = []
        for text_sent in text_sentences:
            sent = self.get_idx_from_sent(text_sent, insert_new_words)
            data.append(sent)
        return self.pad_sentences(data, max_l)

    def pad_sentences(self,data, maxlen, value=0.):
        lengths = [len(s) for s in data]
        nb_samples = len(data)
        if maxlen is None:
            maxlen = np.max(lengths)
        x = (np.ones((nb_samples, maxlen)) * value).astype(np.int)
        for idx, s in enumerate(data):
            if len(s) == 0:
                continue  # empty list was found
            trunc = s[-maxlen:]
            x[idx, -len(trunc):] = trunc
        return x, maxlen
