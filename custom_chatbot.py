import numpy as np
import os 
folder = os.path.dirname(os.path.abspath(__file__))
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
from keras.models import load_model
model = load_model(folder+'/chatbot_model.h5')
import json
import random
intents = json.loads(open(folder+'/intents.json').read())
words = pickle.load(open(folder+'/words.pkl','rb'))
classes = pickle.load(open(folder+'/classes.pkl','rb'))

class chatbot:    
    def clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    def bow(self, sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))
    def predict_class(self,sentence, model):
        # filter out predictions below a threshold
        p = self.bow(sentence, words,show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
    def getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
    def chatbot_response(self, msg):
        ints = self.predict_class(msg, model)
        res = self.getResponse(ints, intents)
        return res
    
#t = 'hi'
#c = chatbot()
#print(c.chatbot_response(t))