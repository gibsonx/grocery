from aux_func import remove_emoji
import os
import re
import string
import pandas as pd
from zhon import hanzi

class DataPrep:
    def __init__(self, base_dir, dataset_name):
        self.dataset_name = dataset_name
        self.dir = base_dir + dataset_name

    def preprocess_tweet_text(tweet):
        tweet = tweet.lower()
        # Remove urls
        tweet = re.sub(r"^#\S+|\s#\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r"^@\S+|\s@\S+", '', tweet)
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        tweet = tweet.translate(str.maketrans('', '', hanzi.punctuation))
        tweet = remove_emoji(tweet)
        return tweet

    def readfile(self,text):
        pd_list = []
        with open(text, 'r') as f:
            for tweet in f.read().splitlines():
                tweet = self.preprocess_tweet_text(tweet)
                pd_list.append(tweet)
        return pd_list

    def readfile_label(self,text):
        pd_list = []
        with open(text, 'r') as f:
            for tweet in f.read().splitlines():
                pd_list.append(int(tweet))
        return pd_list

    def dataframe(self):
        dict_train = {'text': self.readfile(os.path.join(self.dir, "train_text.txt")),
                           'label': self.readfile_label(os.path.join(self.dir, "train_labels.txt"))}
        dict_val = {'text': self.readfile(os.path.join(self.dir, "val_text.txt")),
                         'label': self.readfile_label(os.path.join(self.dir, "val_labels.txt"))}
        dict_test = {'text': self.readfile(os.path.join(self.dir, "test_text.txt")),
                          'label': self.readfile_label(os.path.join(self.dir, "test_labels.txt"))}
        return pd.DataFrame(dict_train), pd.DataFrame(dict_val), pd.DataFrame(dict_test)

    def dataframe_merge(self):
        return pd.concat(self.dataframe)

    def binary_split(self):
        pos = self.dataframe_merge[self.dataframe_merge['label'] == 0]
        neg = self.dataframe_merge[self.dataframe_merge['label'] == 1]
        return pos, neg
