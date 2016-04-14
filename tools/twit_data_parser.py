import csv
import numpy as np
import re
import io
from . import prepare_corpus

class TweetDataParser:
    def __init__(self):
        self.lastindex = 0
        self.data = []
        self.sentiments = []
        self.last_word_index = 0
        self.someOneRegex = re.compile(r"@\S+")
        self.charRegex = re.compile(r'["\.\!\)\(=,:\?\n]')
        self.urlRegex = re.compile(r'http\S+')
        self.smileRegex = re.compile(r'(XD)|(xD)|(:D)|(=\)+)|(\){2,})')
        self.wordcount = 0
        pass

    def add_csv_file(self, fileName, sentiment,verbose=1):
        with io.open(fileName, 'r', encoding='utf-8') as csvPositive:
            reader = csv.reader(csvPositive, delimiter=';')
            for row in reader:
                if len(row[3]) > 10:
                    rowText = self.someOneRegex.sub('@someone', row[3].replace(' - ', ''))
                    rowText = prepare_corpus.process_text(rowText)
                    if(verbose > 0):
                        self.wordcount += len(rowText.split())
                        if(self.wordcount % 10000 == 0):
                            print('Total word counts: %dK' % (self.wordcount/1000))
                    self.data.append(rowText)
                    self.sentiments.append(sentiment)
                    self.lastindex += 1



