import numpy as np
import pandas as pd
import string

from flask import Flask, request, jsonify, render_template
import pickle

stopwordsList = pickle.load(open('stopwords.list', 'rb'))


def textPreprocessor(featureRecord):
    #a.Remove Punctuation
    removePunctuation = [char for char in featureRecord if char not in string.punctuation]
    sentences = ''.join(removePunctuation)
    
    #b.Convert Sentences to Words
    words = sentences.split(" ")
    
    #c. Normalize
    wordNormalized = [word.lower() for word in words]
    
    #d. Remove Stopwords
    finalWords = [word for word in wordNormalized if word not in stopwordsList]
    
    return finalWords



model = pickle.load(open('model.mdl', 'rb'))
tfIdfObject = pickle.load(open('tfIdfObject.tfidf','rb'))
finalWordVocab= pickle.load(open('finalWordVocab.bow','rb'))

app = Flask(__name__)





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
 
    
def predict():
    '''
    For rendering results on HTML GUI
    '''
    SMSInput = request.form['SMS']
    
    #1. Preprocess

    preProcessedFeatures = textPreprocessor(SMSInput)

    #2. BOW transformation
    bowFeature = finalWordVocab.transform(preProcessedFeatures)
   
    #3. TFIDF transformation

    tfIdfFeature = tfIdfObject.transform(bowFeature)
	
    #4. Model Predict

    predLabel = model.predict(tfIdfFeature)[0] 

    return render_template('index.html', prediction_text=' The SMS entered is a {}'.format(predLabel))

if __name__ == "__main__":
    def textPreprocessor(featureRecord):
        #a.Remove Punctuation
        removePunctuation = [char for char in featureRecord if char not in string.punctuation]
        sentences = ''.join(removePunctuation)
    
        #b.Convert Sentences to Words
        words = sentences.split(" ")
    
        #c. Normalize
        wordNormalized = [word.lower() for word in words]
    
        #d. Remove Stopwords
        finalWords = [word for word in wordNormalized if word not in stopwordsList]
    
        return finalWords

    app.run(debug=True)
