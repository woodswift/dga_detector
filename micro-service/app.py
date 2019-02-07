import flask
from flask import Flask, render_template, request

# from redis import Redis, RedisError

import numpy as np

import tensorflow as tf

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

import keras.models
from keras.models import model_from_json

import os
import socket
import pickle


# # Connect to Redis
# redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

# init flask app
app = Flask(__name__)

# deserialize tokenizer and model
def init():
    with open('./models/tokenizer.pickle', 'rb') as pickle_file:
        loaded_tokenizer = pickle.load(pickle_file)
    print("Loaded Tokenizer Successfully!!!")

    with open('./models/gru_clf.json','r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights("./models/gru_clf.h5")
    print("Loaded Model Successfully!!!")

    # compile loaded model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()

    return loaded_tokenizer, loaded_model, graph

# utility function to process INPUT DOMAIN with padding
MAX_SEQ_LENGTH = 40

def prepare_data(sequence, padding='pre'):
    data = tok.texts_to_sequences([sequence,])
    data = pad_sequences(data, maxlen=MAX_SEQ_LENGTH, padding=padding)

    return data

# init tokenizer and model
global tok, model, graph
tok, model, graph = init()


@app.route('/')
def index():
    # try:
    #     visits = redis.incr("counter")
    # except RedisError:
    #     visits = "<i>cannot connect to Redis, counter disabled</i>"

    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {"success": False}

    # get the request parameters
    params = flask.request.json
    if params == None:
        params = flask.request.args

    # if parameters are found, echo the domain parameter
    if params != None:
        data["domain"] = params.get("domain")
        x = prepare_data(data["domain"])
        with graph.as_default():
            y = model.predict(x)
            y_prob = y[:,1]
            y_label = np.argmax(y, axis=-1)
            data["probability_to_be_dga"] = str(y_prob[-1])
            if y_label[-1]:
                data["label"] = "dga"
            else:
                data["label"] = "legit"
            data["success"] = True

    # return a response in json format
    return flask.jsonify(data)


if __name__ == "__main__":

    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port)
