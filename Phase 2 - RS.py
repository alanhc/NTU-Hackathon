#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import math
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Activation, Input
from keras.optimizers import Adam
import tensorflow as tf

_sum = [15, 20, 25, 30, 35, 40]
n = 4
fps = 30
samples = []
labels = []

def generate_distribution(num_of_frame, normalize=False):
    num_of_frame = num_of_frame * fps + 1
    values = range(0, num_of_frame)
    probs = [1.0 / num_of_frame] * num_of_frame

    for idx, prob in enumerate(probs):
        if idx > 3 and idx < 20:
            probs[idx] = probs[idx] * (1 + math.log(idx + 1))
        if idx > 20 and idx < 40:
            probs[idx] = probs[idx] * (1 + math.log((40 - idx) + 1))

    probs = [p / sum(probs) for p in probs]
    if normalize:
        s =  np.random.choice(values, 4, p=probs)
        sample = [i/sum(s) for i in s]
        sample = np.array(sample)        
    else:
        sample =  np.random.choice(values, 4, p=probs)
    label = np.zeros(sample.shape)
    label[sample.argsort()[-2:][::-1]] = 1.0
    
    return sample, label

for n in range(100):
    for i in _sum:
        sample, label = generate_distribution(i, normalize=True)
        samples.append(sample)
        labels.append(label)
samples = np.array(samples)
labels = np.array(labels)
print(samples.shape)
print(labels.shape)

x_in = Input(shape=(4,))
x = Dense(10)(x_in)
x = Activation('relu')(x)
x = Dense(10)(x)
x = Activation('relu')(x)
x_dense = Dense(4)(x)
x_out = Activation('sigmoid')(x_dense)

model = Model(inputs=x_in, outputs=x_out)
print(model.summary())
adam = Adam(lr = 0.0005)
model.compile(optimizer = adam, loss = "mean_squared_error", metrics=['accuracy'])

history = model.fit(x = samples,
                    y = labels,
                    epochs=500,
                    batch_size=50, 
                    verbose=1)

model.save("rs_model.h5")

np.set_printoptions(suppress=True)
pred = model.predict(np.array([100, 40, 83, 33]).reshape((1, 4)))
print(pred)
