import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import re

data = pd.read_csv("C:/Users/HP/Desktop/New folder/train.csv",encoding= 'unicode_escape')

# Keeping only the neccessary columns
data = data[['selected_text','sentiment']]
def preProcess_data(text):
   text = str(text).lower()
   new_text = re.sub('[^a-zA-z0-9\s]','',text)
   new_text = re.sub('rt', '', new_text)
   return new_text

data['selected_text'] = data['selected_text'].apply(preProcess_data)

max_features = 2000

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['selected_text'].values)
X = tokenizer.texts_to_sequences(data['selected_text'].values)
X = pad_sequences(X, 28) 
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30)

embed_dim = 28
lstm_out = 196

early_stopping = EarlyStopping(monitor='val_loss',patience=10)

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(lstm_out, dropout=0.5, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(128,recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

batch_size = 256

model.fit(X_train, Y_train, epochs = 100, batch_size=batch_size, validation_data=(X_test, Y_test),callbacks=[early_stopping])
model.save('sentiment.h5')
