'''

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
        return text
    except Exception as e:
        return str(e)

# Load the Multinomial Naive Bayes model
model_path = 'model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
vectorizer_path = 'vectorizer.pkl'
with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Load training data
training_data_path = 'training_data.csv'
train_df = pd.read_csv(training_data_path)
X_train = tfidf.transform(train_df['text']).toarray()
y_train = train_df['target'].values

# Additional print statements for debugging
print("Loaded Model Parameters:", model.get_params())

st.title("Legal Document Classifier")

input_url = st.text_input("Enter the link")

if st.button('Classify'):
    if input_url:
        # Fetch text from the URL
        fetched_text = fetch_text_from_url(input_url)

        # 1. preprocess
        transformed_sms = transform_text(fetched_text)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Before prediction
        st.write("Transformed SMS:", transformed_sms)
        st.write("Vector Input:", vector_input)
        # 3. predict
        # Fit the model with appropriate data before prediction
        model.fit(X_train, y_train)
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Not Cybercrime")
        else:
            st.header("Cybercrime")
    else:
        st.warning("Please enter a valid URL.")
'''

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import requests
from bs4 import BeautifulSoup
import os
print(os.path.isfile('model.pkl'))


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
        return text
    except Exception as e:
        return str(e)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))
print(type(model))
print("Model Parameters:", model.get_params())



st.title("Legal Document Classifier")

input_url = st.text_input("Enter the link")

if st.button('Classify'):
    if input_url:
        # Fetch text from the URL
        fetched_text = fetch_text_from_url(input_url)

        # 1. preprocess
        transformed_sms = transform_text(fetched_text)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Not Cybercrime")
        else:
            st.header("Cybercrime")
    else:
        st.warning("Please enter a valid URL.")