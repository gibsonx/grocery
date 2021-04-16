import os,pathlib,re,nltk,pickle,string
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow.keras import Input, layers, losses, preprocessing, utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers.core import Dense,Dropout
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# ML Libraries
from sklearn.metrics import accuracy_score,confusion_matrix
from zhon import hanzi
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import csv
import tokenization
from focal_loss import BinaryFocalLoss
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from transformers import TFDistilBertModel,DistilBertTokenizer
try:
    %tensorflow_version 2.x
except Exception:
    pass
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def imbalance_under_sampling(dfname):
  df_label_a = dfname[dfname['label'] == 1]
  df_label_b = dfname[dfname['label'] == 0]
  if df_label_a.shape[0] > df_label_b.shape[0]:
  # count_class_0, count_class_1 = dfname.label.value_counts()
    df_label_a = df_label_a.sample(df_label_b.shape[0],random_state=1)
    df = pd.concat([df_label_b, df_label_a], axis=0)
    print('label 0 is more',df.label.value_counts())
    return df
  else:
    df_label_b = df_label_b.sample(df_label_a.shape[0],random_state=1)
    df = pd.concat([df_label_b, df_label_a], axis=0)
    print('label 1 is more',df.label.value_counts())
    return df

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)