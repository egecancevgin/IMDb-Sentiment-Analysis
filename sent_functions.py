import tensorflow as tf
import os
import numpy as np
from keras.datasets import imdb
from tensorflow.keras import utils
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


def encode_text(text, word_index, maxlen):
    """
    Performs the text encoding
    :param text: The text to be encoded
    :param word_index: A dictionary containing the word to index mapping
    :param maxlen: The maximum length of the encoded text
    :return: The encoded text as a numpy array
    Complexity: O(n), S(n)
    """
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return utils.pad_sequences([tokens], maxlen)[0]


def decode_integers(integers, reverse_word_index):
    """
    Decodes the integers
    :param integers: A list of integers to be decoded
    :param reverse_word_index: A dictionary containing the index to word mapping
    :return: The decoded text
    Complexity: O(n), S(n)
    """
    PAD = 0
    text = ''

    # Traversing the list
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + ' '

    return text[:-1]


def build_model(hp):
    """
    Builds the model, to be used for the Hyperparameter Tuning operation
    :param hp: The hyperparameters to be used to build the model
    :return: The built model
    Complexity: O(n), S(n)
    """
    VOCAB_SIZE = 88584

    # Creating the model with an Embedding layer + LSTM + Dense layer
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, hp.Int('embedding_units', min_value=16, max_value=128, step=16)),
        tf.keras.layers.LSTM(hp.Int('lstm_units', min_value=16, max_value=128, step=16)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compiling the model with binary cross-entropy loss function and RMSProp
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    # Returning the model
    return model


def predict(text, model, word_index, maxlen):
    """
    Making a prediction using the trained model
    :param text: Str, the text to be classified
    :param model: tf.keras.Model, the trained model
    :param word_index: Dict, the word index of the IMDB dataset
    :param maxlen: Int, the maximum length of the text sequence
    :return: Float, the predicted sentiment score between 0 and 1
    Complexity: O(1), S(1)
    """

    # First, encoding the input text
    encoded_text = encode_text(text, word_index, maxlen)

    # Setting the first element of the array the encoded text, with maximum length size
    prediction = np.zeros((1, 250))
    prediction[0] = encoded_text

    # Using the trained model to make the prediction
    result = model.predict(prediction)
    return result[0]
