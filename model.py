#import the Required libraries
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

#load the training data
df = pd.read_csv("train.csv")

X = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 200000 # number of words in the vocab

#text preprocessing 
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

#saving the text vectorization for future use
import pickle
pickle.dump({'config': vectorizer.get_config(),
             'weights': vectorizer.get_weights()}
            , open("tv_layer.pkl", "wb"))

#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(8) # helps bottlenecks

#Splitting the dataset into train,test & val
train_data = dataset.take(int(len(dataset)*.7))
val_data = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test_data = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

#Model
model = Sequential()
# Create the embedding layer
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam',metrics=['accuracy'])
model.summary()

#training the model
history = model.fit(train_data, epochs=5, validation_data=val_data)

#testing the model
model.evaluate(test_data)

#plotting the loss curve

plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

#save the model 
model.save('cmodel.h5')
