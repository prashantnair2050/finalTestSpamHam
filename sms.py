import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pickle
import string
import dill as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def textPreprocessor(featureRecord):
    stopwordsList = stopwords.words('english')
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

if __name__ == '__main__':
    data = pd.read_csv('SMSSpamCollection', sep="\t", names=['label','message'])
    #Create feature and label set
    stopwordsList = stopwords.words('english')
    features = data.iloc[:,[1]].values
    label = data.iloc[:,0].values
    #Initialize Count Vectorizer with our custom preprocessing function
    wordVector = CountVectorizer(analyzer=textPreprocessor)
    #Build Vocab
    finalWordVocab = wordVector.fit(features)
    #Create BOW
    bagOfWords = finalWordVocab.transform(features)
    #Apply TFIDF on BOW and Calc All Values (TF and IDF)
    tfIdfObject = TfidfTransformer().fit(bagOfWords)
    #Transform data (Calc Weights )
    finalFeature = tfIdfObject.transform(bagOfWords)
    
    #Create Train Test Split
    X_train,X_test,y_train,y_test = train_test_split(finalFeature,
                                                label,
                                                test_size=0.2,
                                                random_state=6)
    
    #Build Model
    model = LogisticRegression()
    model.fit(X_train,y_train)
    print("Model trained successfully ...")
    #Check Quality of Model
    #1. Check whetehr model is generalized or not
    print(model.score(X_train,y_train))
    print(model.score(X_test,y_test))
    print("test")
    
    
    pkl.dump(stopwordsList,open('stopwords.list','wb'))
    pkl.dump(textPreprocessor,open('textPreprocessor.fn','wb'))
    pkl.dump(finalWordVocab,open('finalWordVocab.bow','wb'))
    pkl.dump(tfIdfObject,open('tfIdfObject.tfidf','wb'))
    pkl.dump(model,open('model.mdl','wb'))