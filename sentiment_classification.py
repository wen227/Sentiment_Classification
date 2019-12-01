# -*- coding: utf-8 -*-
"""
IMDB dataset sentiment classification
author:wen227
github:https://github.com/wen227/Sentiment_Classification
reference:
1.https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
  This page presents a simple example of LSTM neural network model
2.https://zhuanlan.zhihu.com/p/49271699 This article tells the history of word embedding
3.https://my.oschina.net/u/3800567/blog/2887156 This article shows the way to plot results
In order to set hyperparameters ,I also look for many articles which are listed in my experimental report.
"""
# Importing the libraries
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout, LSTM, Dense
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.models import load_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # set Graphviz path


def show_train_history(train_history):
    print(train_history.history.keys())
    print(train_history.epoch)
    plt.plot(train_history.history['accuracy'], label='accuracy')
    plt.plot(train_history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("acc.png")
    plt.show()

    plt.plot(train_history.history['loss'], label='loss')
    plt.plot(train_history.history['val_loss'], label='val_loss')
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


if __name__ == "__main__":
    # Hyperparameter settings
    max_features = 3000
    max_review_length = 200
    embedding_vecor_length = 32
    _dropout = 0.3  # 0.3
    _epochs = 10  # 10
    # Load IMDB dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path='imdb.npz',
                                                          num_words=max_features,  # keep top 3000
                                                          skip_top=5,  # because top 5 word like 'I' may be useless
                                                          maxlen=None,
                                                          seed=35,  # random seed
                                                          start_char=1,  # seq start from 1
                                                          oov_char=2,
                                                          index_from=3)
    # Truncate and pad input sequences into same size
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # Create the model
    model = Sequential()
    model.add(Embedding(max_features, embedding_vecor_length, input_length=max_review_length))  # Embedding layer
    model.add(Dropout(_dropout))      # Droupout layer is used for preventing overfitting
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))  # 1D convolution layer
    model.add(MaxPooling1D(pool_size=2))  # MaxPooling layer
    model.add(LSTM(128, dropout=_dropout, recurrent_dropout=_dropout))  # LSTM
    model.add(Dropout(_dropout))      # Droupout layer
    model.add(Dense(1, activation='sigmoid'))  # Project onto a single unit output layer, and squash it with a sigmoid
    # Configures the model for training
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # Summary
    print(model.summary())
    # Train and show the process
    train_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=_epochs, batch_size=32)
    show_train_history(train_history)
    # Save model
    model.save(filepath="model.h5")
    # # 加载模型
    # model = load_model('model.h5')
    # show_train_history(model)
    # 模型图像绘制
    plot_model(model=model, to_file="model.png",
               show_layer_names=True, show_shapes=True)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


