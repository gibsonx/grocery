import re
import pandas as pd
import os,string
import pandas as pd
from zhon import hanzi
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, layers, losses, preprocessing, utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from zhon import hanzi
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import csv
from focal_loss import BinaryFocalLoss

class DataPrep:
    def __init__(self, base_dir, dataset_name):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.dir = self.base_dir + dataset_name

    def remove_emoji(self, string):
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

    def preprocess_tweet_text(self,tweet):
        tweet = tweet.lower()
        # Remove urls
        tweet = re.sub(r"^#\S+|\s#\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r"^@\S+|\s@\S+", '', tweet)
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        tweet = tweet.translate(str.maketrans('', '', hanzi.punctuation))
        tweet = self.remove_emoji(tweet)
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
        return pd.concat(self.dataframe()).reset_index(drop=True)

    def binary_split(self):
        df = self.dataframe_merge()
        pos = df[df['label'] == 0].reset_index(drop=True)
        neg = df[df['label'] == 1].reset_index(drop=True)
        return pos, neg

def imbalance_under_sampling(dfname):
  df_label_a = dfname[dfname['label'] == 1]
  df_label_b = dfname[dfname['label'] == 0]
  if df_label_a.shape[0] > df_label_b.shape[0]:
    df_label_a = df_label_a.sample(df_label_b.shape[0],random_state=1)
    df = pd.concat([df_label_b, df_label_a], axis=0)
    print('label 0 is more',df.label.value_counts())
    return df
  else:
    df_label_b = df_label_b.sample(df_label_a.shape[0],random_state=1)
    df = pd.concat([df_label_b, df_label_a], axis=0)
    print('label 1 is more',df.label.value_counts())
    return df

def create_xlnet(model_name):
    """ Creates the model. It is composed of the XLNet main block and then
    a classification head its added
    """
    # Define token ids as inputs
    word_ids = tf.keras.Input(shape=(120,), name='word_ids', dtype='int32')
    word_attention = tf.keras.Input(shape=(120,), name='word_attention', dtype='int32')
    # word_seq = tf.keras.Input(shape=(120,), name='word_seq', dtype='int32')

    # Call XLNet model
    xlnet_encodings = model_name([word_ids,word_attention])[0]

    # CLASSIFICATION HEAD
    # Collect last step from last hidden state (CLS)
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    # Apply dropout for regularization
    dense = tf.keras.layers.Dense(32, activation='relu', name='encoding')(doc_encoding)
    drop = tf.keras.layers.Dropout(0.1)(dense)
    # Final output
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(drop)

    # Compile model
    model = tf.keras.Model(inputs=[word_ids,word_attention], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

    # Compile model
    model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

def get_inputs(tweets, tokenizer, max_len=120):
    """ Gets tensors from text using the tokenizer provided"""
    inps = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in tweets]
    inp_tok = np.array([a['input_ids'] for a in inps])
    ids = np.array([a['attention_mask'] for a in inps])
    return inp_tok, ids

def plot_metrics(pred, true_labels):
    """Plots a ROC curve with the accuracy and the AUC"""
    acc = accuracy_score(true_labels, np.array(pred.flatten() >= .5, dtype='int'))
    fpr, tpr, thresholds = roc_curve(true_labels, pred)
    auc = roc_auc_score(true_labels, pred)

    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.plot(fpr, tpr, color='red')
    ax.plot([0,1], [0,1], color='black', linestyle='--')
    ax.set_title(f"AUC: {auc}\nACC: {acc}");
    return fig

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy

  if cmap is None:
      cmap = plt.get_cmap('Blues')

  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()

  if target_names is not None:
      tick_marks = np.arange(len(target_names))
      plt.xticks(tick_marks, target_names, rotation=45)
      plt.yticks(tick_marks, target_names)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


  thresh = cm.max() / 1.5 if normalize else cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      if normalize:
          plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
      else:
          plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
  plt.show()