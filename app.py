import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

def preProcess_data(text): #cleaning the data
    
    text = str(text).lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

app = FastAPI()

data = pd.read_csv("C:/Users/HP/Desktop/New folder/train.csv",encoding= 'unicode_escape')
tokenizer = Tokenizer(num_words=25000, split=' ')
tokenizer.fit_on_texts(str(data['text'].values))



def my_pipeline(text): #pipeline
  text_new = preProcess_data(text)
  X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
  X = pad_sequences(X, maxlen=28)
  return X




@app.get('/', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="28" name="text" value="Text Emotion to be tested"/>  
    <input type="submit"/> 
    </form>'''



@app.post('/') #prediction on data
def predict(text:str = Form(...)): #input is from forms
    clean_text = my_pipeline(text) #cleaning and preprocessing of the texts
    loaded_model = tf.keras.models.load_model('sentiment.h5') #loading the saved model
    predictions = loaded_model.predict(clean_text) #making predictions
    sentiment = int(np.argmax(predictions)) #index of maximum prediction
    probability = max(predictions.tolist()[0]) #probability of maximum prediction
    if sentiment==2: #assigning appropriate name to prediction
        t_sentiment = 'negative'
    elif sentiment==1:
        t_sentiment = 'neutral'
    elif sentiment==0:
        t_sentiment='positive'
    
    return { #returning a dictionary as endpoint
        "ACTUAL SENTENCE": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability": probability
    }