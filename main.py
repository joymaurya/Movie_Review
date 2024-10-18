from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from create_model import Model
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
pytorch_model=Model()
pytorch_model.load_state_dict(torch.load("movie_review_pytorch_model.pth",map_location=torch.device('cpu')))

word_to_index=imdb.get_word_index()

index_to_word={value:key for key,value in word_to_index.items()} 

def preprocess(text):
    words=text.lower().split(" ")
    text=[word_to_index.get(word,2) +3 for word in words]
    text=pad_sequences([text],maxlen=500)
    return text

def predict(text):
    text=preprocess(text)
    text=torch.Tensor(text).type(torch.int32)
    logits=pytorch_model(text)
    prediction=torch.sigmoid(logits)
    sentiment="Positive" if prediction.item()>0.5 else "negative"
    return sentiment,prediction.item()

st.title("Movie Review Sentiment")
review=st.text_input("Write Your Review")
sentiment,score=predict(review)

st.write(f"Review is {sentiment} with score {score}")
