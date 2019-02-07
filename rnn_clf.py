import numpy as np

import pandas as pd
from pandas.api.types import CategoricalDtype

import pickle

from collections import Counter
from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score

import tensorflow as tf

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Activation, LSTM, GRU, Bidirectional, Embedding, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model, load_model
from keras.layers.merge import concatenate

from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, optimizers, constraints


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text line by line
    text = file.readlines()
    # close the file
    file.close()

    return text

def pre_process(record):
    # trimming
    record = str(record).strip()
    # split by '\x01'
    record = record.split('\x01')
    # lowercase label
    record = [record[0], record[1], record[2].lower()]

    return record

def split_data(df, test_perc):
    df_train, df_test = train_test_split(df, test_size=test_perc, random_state=55)

    return df_train, df_test

def load_data(sequences, labels, padding='pre'):
    data = tok.texts_to_sequences(sequences)
    data = pad_sequences(data, maxlen=MAX_SEQ_LENGTH, padding=padding)

    return data, np.asarray(labels).astype('int32')

def transform_to_cate(y):
    return np_utils.to_categorical(y, 2)

def transform_to_class(vec):
    return vec.argmax(axis=-1)

def get_pos_prob(y_predicted):
    return y_predicted[:,1]

def print_result(data_name, y_truth, y_predicted):
    print("\nPerformance of " + data_name + " dataset:\n")
    print(classification_report(y_truth, y_predicted))

def get_result(data_name, y_truth, y_prob, y_predicted):
    print_result(data_name, y_truth, y_predicted)

    auc = roc_auc_score(y_truth, y_prob, sample_weight=None)
    print("AUC: %s" % auc)
    return auc


# train : dev : test: unused = 8 : 1 : 1
DEV_PERC = 1.0/10
TEST_PERC = 1.0/9

# lowercase, uppercase, digit, '.', '-'
MAX_NB_CHARS = 26 * 2 + 10 + 2
# 99% percentile: 41
MAX_SEQ_LENGTH = 40
EMBEDDING_DIM = 100
UNIT = 128


if __name__ == "__main__":

    # ------ data preparation ------ #
    # load doc
    filename = "./dga-dataset.txt"
    raw_doc = load_doc(filename)
    print(len(raw_doc))
    print('')

    # csv-like file
    # 3 columns: domain, origin, label
    raw_doc = list(filter(lambda x: x.count('\x01') == 2, raw_doc))
    print(len(raw_doc))
    print('')

    # pre-process
    raw_doc = list(map(pre_process, raw_doc))
    columns = ['domain', 'origin', 'label']

    # form pd.DataFrame
    df = pd.DataFrame.from_records(raw_doc, columns=columns)
    print(df.count())
    print('')

    # filter by two classes: legit, dga
    df = df.loc[df['label'].isin(['legit', 'dga']), :]
    print(df.count())
    print('')
    print(df['origin'].value_counts())
    print('')
    print(df['label'].value_counts())
    print('')

    # legit: 0, dga: 1
    df['label'].replace(to_replace={'legit': 0, 'dga': 1}, inplace=True)
    df['label'] = df['label'].astype(CategoricalDtype([0, 1]))

    # check label distribution
    print("label distribution in df:")
    print(df['label'].value_counts())
    print(df['label'].value_counts(normalize=True))
    print('')

    # MAX_LENGTH STAT
    df['length'] = df['domain'].apply(lambda x: len(x))
    # mean: 20
    # median: 18
    # max: 60
    # 90%: 33
    # 95%: 35
    # 97%: 37
    # 99%: 41
    print(df['length'].describe([.25, .5, .75, .9, .95, .97, .99]))
    print('')

    # MAX_NB_CHARS STAT
    char_vocab = Counter()
    for domain in df['domain'].tolist():
        char_vocab += Counter(domain)
    # 38 char found

    # form train, dev, test datasets
    df_train, df_dev = split_data(df, DEV_PERC)
    df_train, df_test = split_data(df_train, TEST_PERC)

    # confirm label distribution
    print("label distribution in df_train:")
    print(df_train['label'].value_counts())
    print(df_train['label'].value_counts(normalize=True))
    print('')
    print("label distribution in df_dev:")
    print(df_dev['label'].value_counts())
    print(df_dev['label'].value_counts(normalize=True))
    print('')
    print("label distribution in df_test:")
    print(df_test['label'].value_counts())
    print(df_test['label'].value_counts(normalize=True))
    print('')

    # ------ model training ------ #
    texts = df_train['domain'].tolist()

    tok = Tokenizer(num_words=MAX_NB_CHARS, filters=None, lower=False, char_level=True)
    tok.fit_on_texts(texts)

    char_index = tok.word_index
    print("Total %s unique chars." % "{:,}".format(len(char_index)))
    print('')

    # save tokenizer
    with open("./models/tokenizer.pickle", "wb") as pickle_file:
        pickle.dump(tok, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # prepare sequence input with pre padding
    X_train, y_train = load_data(df_train['domain'].tolist(), df_train['label'].tolist(), padding='pre')
    X_dev, y_dev = load_data(df_dev['domain'].tolist(), df_dev['label'].tolist(), padding='pre')
    X_test, y_test = load_data(df_test['domain'].tolist(), df_test['label'].tolist(), padding='pre')

    # one-hot labels
    y_train = transform_to_cate(y_train)
    y_dev = transform_to_cate(y_dev)
    y_test = transform_to_cate(y_test)

    # confirm shape
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_dev shape:", X_dev.shape)
    print("y_dev shape:", y_dev.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print('')

    # RNN model
    embedding_layer = Embedding(input_dim = MAX_NB_CHARS + 1, \
                                output_dim = EMBEDDING_DIM, \
                                input_length = MAX_SEQ_LENGTH)

    seq_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(seq_input)
    layer_1 = GRU(UNIT)(embedded_sequences)

    preds = Dense(2, activation='softmax')(layer_1)

    model = Model(seq_input, preds)

    model.compile(loss='categorical_crossentropy',\
                  optimizer='adam',\
                  metrics=['accuracy'])

    model.summary()

    name = "gru_vob{}_len{}_unit{}_{}".format(MAX_NB_CHARS, \
                                              MAX_SEQ_LENGTH, \
                                              UNIT, \
                                              int(time()))
    print(name)
    print('')

    model.fit(X_train, y_train, batch_size=128, epochs=10, \
              validation_data=(X_dev, y_dev), \
              callbacks=[TensorBoard(log_dir="./logs/{}".format(name))])

    # TensorBoard(log_dir="./logs/{}".format(name))
    # EarlyStopping(monitor='val_loss', min_delta=0.0001)

    # save model
    # serialize model to JSON
    model_json = model.to_json()
    with open("./models/gru_clf.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights
    model.save_weights("./models/gru_clf.h5")

    # ------ performance evaluation ------ #
    yy_train = model.predict(X_train)
    yy_dev = model.predict(X_dev)
    yy_test = model.predict(X_test)

    # get the probability value of being DGA
    yy_train_prob = get_pos_prob(yy_train)
    yy_dev_prob = get_pos_prob(yy_dev)
    yy_test_prob = get_pos_prob(yy_test)

    # print performance report
    train_performance = get_result("train", transform_to_class(y_train), yy_train_prob, transform_to_class(yy_train))
    dev_performance = get_result("dev", transform_to_class(y_dev), yy_dev_prob, transform_to_class(yy_dev))
    test_performance = get_result("test", transform_to_class(y_test), yy_test_prob, transform_to_class(yy_test))
