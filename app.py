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
labels = []
docs = []

## loop thru pattern in intents json file
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        #word_tokenize() => split sentences into words (to clean data)
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words]
# remove duplicate and sorted the text
words = sorted(list(set(words)))

labels = sorted(labels)

# create 2 list to stored encoded (0 and 1) for bag of words
training = []
ouput = []

out_empty = [0 for _ in range(len(classes))]

# Decode the text to 0 and 1 to bag
for x, doc in enumerate(doc_x):
    bag = []
    wrds = [stremmer.stem(w) for w in doc]

    for w in words:
        # the word existed in the curr pattern that we looping thru
        if w in wrds: 
            bag.append(1)
        else: 
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

#finish cleanning the json data 
training = numpy.array(training)
output = np.array(output)


