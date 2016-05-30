import os
import json
import io
import re
import string
from bs4 import BeautifulSoup
from lxml import etree
import csv
import sys
import pandas as pd
punc = '[\!#$%&()*+,-./:;<=>?\[\]^_`{|}~\"]'
punc_regex = re.compile(punc)
space_regex = re.compile('\s+')
article_regex = re.compile(r'^.*\nTEXT\:', re.M | re.DOTALL)
digits_regex = re.compile('\d+')
someOneRegex = re.compile(r"@\S+\s")
urlfinder = re.compile("https?:\/\/\S+")


def process_text(text,soup=True):
    if soup:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
    text = digits_regex.sub(u'0', text.lower())
    text = urlfinder.sub(u' ReplacedUrl ', text)
    text = punc_regex.sub(u' ', text)
    text = space_regex.sub(u' ', text)
    return text


def process_json(filename):
    with io.open(filename, 'r', encoding='utf8') as data_file:
        data = json.load(data_file)
        comments = []
        for c in data['comments']:
            comments.append(c["body"])
        text = u''.join(comments)
        return process_text(text)


def process_article(filename):
    with io.open(filename, 'r', encoding='utf8') as data_file:
        article = data_file.read()
        search_text = article_regex.sub(u'', article)
        text = process_text(search_text)
        return text


def process_large_kfu_xml(w_file, filename):
    words_count = 0
    context = etree.iterparse(filename, encoding='utf-8', recover=True, events=('start','end'), tag='column')
    for event, element in context:
        if event == 'end':
            if element.attrib['name'] == 'txt' and element.text is not None:
                text = process_text(element.text)
                if len(text) > 50:
                    words_count += len(text.split())
                    w_file.write(text + '\n')
                element.clear()
            elif element.tag == 'column':
                element.clear()
    return words_count

def process_wikipedia_corpora(w_file, dir, verbose=1):
    tagregex = re.compile(r'<doc.*?>(.*?)</doc>', re.M | re.DOTALL)
    all_words = 0
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if verbose:
                print('Words count: {0}\r'.format(all_words))
            fn = os.path.join(path, name)
            with io.open(fn, 'r') as f:
                full_text = f.read()
                mathces = tagregex.findall(full_text)
                for element in mathces:
                    if element is not None:
                        text = process_text(element)
                        if len(text) > 50:
                            all_words += len(text.split())
                            w_file.write(text + '\n')

def process_twitter_csv(w_file, filename, col,delimiter=','):
    words_count = 0
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        for row in csv_reader:
            try:
                rowText = row[col].decode('utf-8')
                if len(rowText) > 10:
                    rowText = someOneRegex.sub('@someone ', rowText)
                    rowText = process_text(rowText, soup=False)
                    words_count += len(rowText.split())
                    w_file.write(rowText)
            except:
                rowText = None
    return words_count


def start_dir_process(w_file, dir, csv_col, verbose=1):
    if verbose:
        print 'start', dir
    all_words = 0
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if verbose:
                print('Words count: {0}\r'.format(all_words))
            file_text = None
            fn = os.path.join(path, name)
            if fn.endswith('.json'):
                file_text = process_json(fn)
            elif fn.endswith('.html') or fn.endswith('.txt'):
                file_text = process_article(fn)
            elif name == 'rus_kfu.xml':
                all_words += process_large_kfu_xml(w_file, fn)
                continue
            elif fn.endswith('.csv'):
                all_words += process_twitter_csv(w_file, fn)
                continue
            else:
                continue
            if file_text is None:
                continue
            all_words += len(file_text.split())
            w_file.write(file_text + '\n')


with io.open('..\\data\\eng_corpora.txt','w',encoding='utf8') as f:
    process_wikipedia_corpora(f,"E:\\NLP\\DATA\\ENG\\raw.en")
    process_twitter_csv(f,"E:\\NLP\\DATA\\ENG\\trainingandtestdata\\training.1600000.processed.noemoticon.csv",5,',')