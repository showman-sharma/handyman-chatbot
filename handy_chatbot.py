import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn import preprocessing
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pickle

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2


lemmatizer = WordNetLemmatizer()

#Predict for multiple samples using batch processing
#Custom tokenizer to remove stopwords and use lemmatization
def customtokenize(str):
    #Split string as tokens
    tokens=nltk.word_tokenize(str)
    #Filter for stopwords
    nostop = list(filter(lambda token: token not in stopwords.words('english'), tokens))
    #Perform lemmatization
    lemmatized=[lemmatizer.lemmatize(word) for word in nostop ]
    return lemmatized


#Loading a Model 
loaded_model = keras.models.load_model("chatbot_model");

#Print Model Summary
# loaded_model.summary()

#Convert input into IF-IDF vector using the same vectorizer model
vectorizer = pickle.load( open( "bot_vectorizer.p", "rb" ) );

encoder = preprocessing.LabelEncoder();
encoder.classes_ = np.load('bot_label_encoder.npy',allow_pickle=True);

#Loading Bot Replies
bot_reply_classes = ['start_convo']+list(encoder.classes_) 
bot_replies = {}
for reply_class in bot_reply_classes:
    with open('bot_replies/'+reply_class+'.txt') as f:
        bot_replies[reply_class] = f.readlines()

#Conversation        

start_state = 'start_convo'
end_state = 'end_convo'

state = start_state
while state!=end_state:
    for line in bot_replies[state]:
        print('Handy: '+line.strip())
        
    user_reply = input('user: ')
    
    predict_tfidf=vectorizer.transform([user_reply]).toarray();
    prediction=np.argmax( loaded_model.predict(predict_tfidf), axis=1 );
    
    state = encoder.inverse_transform(prediction)[0]
#     print(state)
for line in bot_replies[state]:
    print('Handy: '+line.strip())
