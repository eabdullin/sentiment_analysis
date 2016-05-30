# -*- coding: utf-8 -*-
import codecs
import json

import io
import re

import tweepy
import time


def read_keywords(filename):
    keys = []
    with codecs.open(filename, 'r', encoding='utf-8') as file:
        num = 0
        current_keys = []
        for line in file:
            if num != 0 and num % 10 == 0:
                keys.append(current_keys)
                current_keys = []
            else:
                word = line.strip()
                if len(word) > 3:
                    current_keys.append(word)
                    num += 1
    return keys


def load_api(settings_filename):
    with io.open(settings_filename, 'r', encoding='utf-8')  as settings_file:
        data = json.load(settings_file)
        consumer_key = data['consumer_key']  # 'Rc27w3Ua0sVtwC5wUAt86i59l'
        consumer_secret = data['consumer_secret']  # 'rZbwA0oTvaZUds8QfU470qLpoLeWZN7nneRNTdGYkllD5I81A1'
        access_token = data['access_token']  # '525493224-BFgmQM7eWlAWuJk2FRhL1Yk4pqNpFG15PjcPQGd3'
        access_token_secret = data['access_token_secret']  # 'tERVbQ4WdRatfcnu6DMy1FpzJfivQTF2plZbveuz3pnTr'
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        return api


def search(keywords, settings_file, save_to_file, verbose=1):
    if isinstance(keywords, basestring):
        keys = read_keywords(keywords)
    elif isinstance(keywords, list):
        keys = keywords
    else:
        raise ValueError('')

    api = load_api(settings_file)
    space_regex = re.compile('\s+')
    with codecs.open(save_to_file, "w", "utf-8") as writetrain:
        index = 0
        for keys_local in keys:
            query = " OR ".join(keys_local)
            twit_cursor = tweepy.Cursor(
                api.search,
                q=query,
                rpp=100,
                count=5000,
                include_entities=True
            ).items()
            while True:
                try:
                    tweet = twit_cursor.next()
                    text = '%s\t@%s\t%s' % (tweet.id, tweet.user.screen_name, space_regex.sub(u' ',tweet.text));
                    if verbose:
                        print '%d\t%s' % (index, text)
                    index += 1
                    writetrain.write('\n' + text)
                except tweepy.TweepError as e:
                    print 'timeout'
                    time.sleep(60 * 15)
                    continue
                except StopIteration:
                    writetrain.flush()
                    print 'stop iteration'
                    break


search('tweepy_keywords.txt','tweepy_settings.json','searched_kazakh_twits.csv')
