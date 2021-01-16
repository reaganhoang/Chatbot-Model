import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn 
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
lables = []
docs = []

## loop thru pattern in intents json file
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        #word_tokenize() => split sentences into words (to clean data)
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])