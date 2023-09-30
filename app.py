import streamlit as st
import pickle
import string
import sklearn
import numpy
import scipy
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def text_transform(text):
    # Converting into Lower Case
    text = text.lower()

    # String Tokenization
    text = nltk.word_tokenize(text)

    # Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():  # isalnum() - checks wether the character is alphanumeeric or not
            y.append(i)

    # Removing Stopwords and penctuation-> stop words are the words which help us to build sentences (eg:- is, the, and)
    text = y[:]  # ":" -> clones the list y
    y.clear()

    for i in text:
        if i not in stopwords.words("english"):
            y.append(i)

    # Stemming -> converts (eg: Dancing to Danc, Loving to Love, etc)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # returns in the form of string

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # Stem-1: Preprocess
    transformed_sms = text_transform(input_sms)
    # Stem-2: Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Stem-3: Predict
    result = model.predict(vector_input)[0]
    # Stem-4: Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
