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
punc_regex = re.compile('[%s]' % re.escape(string.punctuation))
space_regex = re.compile('\s+')
article_regex = re.compile(r'^.*\nTEXT\:',re.M|re.DOTALL)
digits_regex = re.compile('\d+([\.\,]\d+)?')
date_regex = re.compile(r'\d+\.\d+\.\d+')
someOneRegex = re.compile(r"\s@\S+")

def process_text(text):
    soup = BeautifulSoup(text.lower(), 'html.parser')
    text = soup.get_text()
    text = date_regex.sub(u' REPLACEDDATE ',text)
    text = digits_regex.sub(u' REPLACEDNUMBER ',text)
    text = punc_regex.sub(u' ',text)
    text = space_regex.sub(u' ',text)
    return text

def process_json(filename):
    with io.open(filename, 'r', encoding='utf-8') as data_file:
        data = json.load(data_file)
        comments = []
        for c in data['comments']:
            comments.append(c["body"])
        text = u''.join(comments)
        return process_text(text)

def process_article(filename):
    with io.open(filename, 'r', encoding='utf-8') as data_file:
        article = data_file.read()
        search_text = article_regex.sub(u'',article)
        text = process_text(search_text)
        return text

def process_large_kfu_xml(w_file, filename):
    context = etree.iterparse(filename, encoding='utf-8', recover=True, events=('end',), tag='column')
    for event, element in context:
        if event == 'end':
            if element.attrib['name'] == 'txt' and element.text is not None:
                text = process_text(element.text)
                w_file.write(text+'\n')
                element.clear()
            elif element.tag == 'column':
                element.clear()

def process_twitter_csv(w_file, filename):
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        for row in csv_reader:
            rowText = row[3].decode('utf-8')
            if len(rowText) > 10:
                rowText = someOneRegex.sub('@someone', rowText)
                rowText = process_text(rowText)
                w_file.write(rowText)

def start_dir_process(w_file,dir):
    for path, subdirs, files in os.walk(dir):
        for name in files:
            file_text = None
            fn = os.path.join(path, name)
            if fn.endswith('.json'):
                file_text = process_json(fn)
            elif fn.endswith('.html') or fn.endswith('.txt'):
                file_text = process_article(fn)
            elif name == 'rus_kfu.xml':
                process_large_kfu_xml(w_file, fn)
                continue
            elif fn.endswith('.csv'):
                process_twitter_csv(w_file,fn)
                continue
            else:
                continue
            if file_text is None:
                continue
            w_file.write(file_text+'\n')


