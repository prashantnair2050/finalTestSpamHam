import numpy as np
import pandas as pd
import string
import dill as pkl
import nltk
from nltk.corpus import stopwords
import sms

from flask import Flask, request, jsonify, render_template


#stopwordsList = pickle.load(open('stopwords.list', 'rb'))
#textPreprocessor = pickle.load(open('textPreprocessor.fn','rb'))
#model = pickle.load(open('model.mdl', 'rb'))
#tfIdfObject = pickle.load(open('tfIdfObject.tfidf','rb'))
#finalWordVocab= pickle.load(open('finalWordVocab.bow','rb'))

app = Flask(__name__)





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
 
    
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #stopwordsList = pkl.load(open('stopwords.list', 'rb'))
    #textPreprocessor = pkl.load(open('textPreprocessor.fn','rb'))
    model = pkl.load(open('model.mdl', 'rb'))
    tfIdfObject = pkl.load(open('tfIdfObject.tfidf','rb'))
    finalWordVocab= pkl.load(open('finalWordVocab.bow','rb'))
    
    SMSInput = request.form['SMS']
    
    #1. Preprocess

    preProcessedFeatures = sms.textPreprocessor(SMSInput)

    #2. BOW transformation
    bowFeature = finalWordVocab.transform(preProcessedFeatures)
   
    #3. TFIDF transformation

    tfIdfFeature = tfIdfObject.transform(bowFeature)
	
    #4. Model Predict

    predLabel = model.predict(tfIdfFeature)[0] 

    return render_template('index.html', prediction_text=' The SMS entered is a {}'.format(predLabel))

if __name__ == "__main__":
    app.run(debug=True)
