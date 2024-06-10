# import streamlit as st
# import pickle
# import string
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import requests
# from bs4 import BeautifulSoup
# import os
# from collections import Counter  # Import Counter for word frequency

# print(os.path.isfile('model.pkl'))

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# def fetch_text_from_url(url):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.content, 'html.parser')
#         paragraphs = soup.find_all('p')
#         text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
#         return text
#     except Exception as e:
#         return str(e)

# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))
# print(type(model))
# print("Model Parameters:", model.get_params())

# st.title("Legal Document Classifier")

# input_url = st.text_input("Enter the link")

# if st.button('Classify'):
#     if input_url:
#         # Fetch text from the URL
#         fetched_text = fetch_text_from_url(input_url)

#         # 1. preprocess
#         transformed_sms = transform_text(fetched_text)
#         # 2. vectorize
#         vector_input = tfidf.transform([transformed_sms])
#         # 3. predict
#         result = model.predict(vector_input)[0]

#         # 4. Display result
#         if result == 1:
#             st.header("Not Cybercrime")
#         else:
#             st.header("Cybercrime")

#         # Additional features
#         # Word count
#         word_count = len(transformed_sms.split())
#         # Word frequency
#         words_frequency = Counter(transformed_sms.split())
#         # Check if numbers 66 appear

#         # Display outputs side by side
#         col1, col2 = st.columns(2)

#         # Column 1
#         with col1:
#             st.write(f"Word Count: {word_count}")
#             st.write("Word Frequency:")
#             for word, count in words_frequency.most_common(5):  # Display top 5 words
#                 st.write(f"{word}: {count}")

#         # Column 2
#         with col2:
#             occurrences_66 = words_frequency.get('66', 0)
#             st.write(f"Occurrences of Section '66': {occurrences_66}")
#             occurrences_67 = words_frequency.get('67', 0)
#             st.write(f"Occurrences of Section '67': {occurrences_67}")
#             occurrences_65 = words_frequency.get('65', 0)
#             st.write(f"Occurrences of Section '65': {occurrences_65}")


#     else:
#         st.warning("Please enter a valid URL.")



# # https://indiankanoon.org/doc/56699948/
# # https://indiankanoon.org/doc/1030160/


import streamlit as st
import pickle
import string
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BartForConditionalGeneration, BartConfig
import test5

ps = PorterStemmer() 

# Define the transform_text function
def transform_text(text):
    # Example text transformation logic
    print("Transforming text...")
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Define the summarize function
def summarize(text, summarizer):
    print("Summarizing text...")
    max_len = 1000
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk)[0]["summary_text"]
        summaries.append(summary)
    summary = " ".join(summaries)
    print("Chunk Summarizing complete.")
    while len(summary) > max_len:
        new_chunks = [summary[i:i+max_len] for i in range(0, len(summary), max_len)]
        new_summaries = []
        for chunk in new_chunks:
            new_summary = summarizer(chunk)[0]["summary_text"]
            new_summaries.append(new_summary)
        summary = " ".join(new_summaries)
        print("Final summarizing text....")
    final_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    final_summary = final_summarizer(summary)[0]["summary_text"]
    print("Summarizing complete.")
    return final_summary

# Streamlit interface
st.title("Legal Document Classifier")

input_url = st.text_input("Enter the link")

if st.button('Classify'):
    if input_url:
        response = requests.get(input_url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string
        parts = title.split(' on ')
        heading = parts[0].strip()
        date_str = parts[1].strip() if len(parts) > 1 else ''
        text = ' '.join([p.get_text() for p in soup.find_all(["p", "blockquote", "pre"])])
        transformed_sms = transform_text(text)
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("Not Cybercrime")         
            st.markdown(f"<h5>Heading: {heading}</h5>", unsafe_allow_html=True)
            st.markdown(f"<h6>Date: {date_str}</h6>", unsafe_allow_html=True)
        else:
            st.header("Cybercrime")
            st.markdown(f"<h5>Heading: {heading}</h5>", unsafe_allow_html=True)
            st.markdown(f"<h6>Date: {date_str}</h6>", unsafe_allow_html=True)
            paragraphs = soup.find_all(["p", "blockquote", "pre"])
            (sections, summary) = test5.document_filter(paragraphs)
            df = pd.read_csv('penal_codes.csv')
            df = df.drop('SL no.', axis=1)
            penal_code_list = sections
            penal_code_list = [code.replace("-", "").replace("(", "").replace(")", "").upper() for code in penal_code_list]

# Filter the DataFrame based on the provided list
            filtered_df = df[df['Penal Code Section'].apply(lambda x: str(x) in penal_code_list)]
            st.markdown(f"<h6 style='margin-top: 15px;'>Summary: </h6>", unsafe_allow_html=True)
            st.write(summary)
            st.markdown(f"<h6 style='margin-top: 15px;'>Sections charged: </h6>", unsafe_allow_html=True)
            st.write(filtered_df.to_markdown(index=False))
        # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # summary = summarize(text, summarizer)
        







# edit the summarize text
# output the necessary penal codes
# output the summary
# only summarize if it's cybercrime