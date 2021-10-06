from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from sklearn.base import TransformerMixin
import string
import spacy
from spacy import *

punctuations = string.punctuation
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
from spacy.lang.en import English
parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words and punctuations
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    if(type(text) == float):
        return text
    else:
        # Removing spaces and converting text into lowercase
        return text.strip().lower()

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    y = json.dumps(data)
    y = y.replace('"',"")
    y = y.replace("[","").replace("]","").replace("'","")
    print(y)
    prediction = model.predict([y])
    print(prediction)
    jsonstring =np.array2string(prediction)
    return jsonify(jsonstring)


if __name__ == '__main__':

    infile = open('C:/Pythonworkings/Execute/Textclassify.pkl', 'rb')
    model = p.load(infile)
    app.run(host='0.0.0.0',port='105')