import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_f1_support
def preprocess(data):
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data
def split(data):
    x = np.array(data["title"])
    y = np.array(data["label"])
    return x, y
def vectorize(x):
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    return x
def train(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(xtrain, ytrain)
    return model, xtest, ytest
def evaluate(model, xtest, ytest):
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    prf = precision_recall_f1_support(ytest, ypred, average="weighted")
    return acc, prf
def fakenewsdetection():
    user = st.text_area("Enter Any News Headline: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)

if __name__ == "__main__":
    data = pd.read_csv("news.csv")
    data = preprocess(data)
    x, y = split(data)
    x = vectorize(x)
    model, xtest, ytest = train(x, y)
    acc, prf = evaluate(model, xtest, ytest)
    print("Accuracy: ", acc)
    print("Precision, Recall, F1 Score: ", prf)
    fakenewsdetection()

data = pd.read_csv("news.csv")

x = np.array(data["title"])
y = np.array(data["label"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)

import streamlit as st
st.title("Fake News Detection System")
def fakenewsdetection():
    user = st.text_area("Enter Any News Headline: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
fakenewsdetection()