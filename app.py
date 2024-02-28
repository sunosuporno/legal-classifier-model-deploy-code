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
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, BartForConditionalGeneration, BartConfig

ps = PorterStemmer() 

def transform_text(text):
    # Existing text transformation code 

    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  
    model = pickle.load(open('model.pkl', 'rb'))

    st.title("Legal Document Classifier")

    input_url = st.text_input("Enter the link")

    if st.button('Classify'):
        if input_url:
            # Fetch text 
            response = requests.get(input_url)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            
            # Summarization
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn") 
            summary = summarize(text, summarizer)
            
            # Existing steps
            transformed_sms = transform_text(text) 
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            st.write(result)
            st.write(summary)

def summarize(text, summarizer):
    max_len = 1000
    
    # Split text into chunks
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk)[0]["summary_text"] 
        summaries.append(summary)
        
    # Concatenate summary chunks
    summary = " ".join(summaries)
    
    # Iterate summarization 
    while len(summary) > max_len:
        new_chunks = [summary[i:i+max_len] for i in range(0, len(summary), max_len)]
        new_summaries = []
        for chunk in new_chunks:
            new_summary = summarizer(chunk)[0]["summary_text"]
            new_summaries.append(new_summary)
        summary = " ".join(new_summaries)
        
    # Final summarization
    final_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    final_summary = final_summarizer(summary)[0]["summary_text"]
    
    return final_summary