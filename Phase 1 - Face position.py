#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import BatchNormalization, Activation, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
import tensorflow as tf

train_data = np.load("/home/hsiehch/NTU-Hackathon/trainingData/trainingData.npy")
test_data = np.load("/home/hsiehch/NTU-Hackathon/trainingData/testingData.npy")
train_data = train_data / 255
test_data = test_data / 255

train_label = np.load("/home/hsiehch/NTU-Hackathon/trainingData/trainingLabel.npy")
test_label = np.load("/home/hsiehch/NTU-Hackathon/trainingData/testingLabel.npy")
train_label = np_utils.to_categorical(train_label, 4)
test_label = np_utils.to_categorical(test_label, 4)


x_in = Input(shape=(None, None,3))
x = Conv2D(filters=8, kernel_size=3)(x_in)
x = Activation('relu')(x)
x = Conv2D(filters=16, kernel_size=3)(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x_dense = Dense(4)(x)
x_out = Activation('softmax')(x_dense)

model = Model(inputs=x_in, outputs=x_out)
print(model.summary())
adam = Adam(lr = 0.001)
model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics=['accuracy'])

losses = []
for epoch in range(10):
    start = 0
    end = 27
    epoch_losses = []
    for i in range(10):
        current_loss = model.train_on_batch(train_data[start:end+1], train_label[start:end+1])
        epoch_losses.append(current_loss)
        start += 28
        end += 28
    losses.append(epoch_losses.sum() / (1.0 * len(epoch_losses)))
final_loss = losses.sum() / (1.0 * len(losses))

losses = []
for data, label in zip(test_data, test_label):
    data = data.reshape((1, data.shape[0], data.shape[1], data.shape[2]))
    label = label.reshape((1, label.shape[0]))
    loss = model.evaluate(data, label, )
    losses.append(loss)

