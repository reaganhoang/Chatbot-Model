import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn 
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        #save all list into pickle file
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    ## loop thru pattern in intents json file
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            #word_tokenize() => split sentences into words (to clean data)
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # remove duplicate and sorted the text
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # create 2 list to stored encoded (0 and 1) for bag of words
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # Decode the text to 0 and 1 to bag
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

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
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        #save all list into pickle file
        pickle.dump((words, labels, training), f)


# creat ML model to predict message
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
# connect the neural network 1s layer with 8 neuron that are fully connected
net = tflearn.fully_connected(net, 8)
# connect the neural network 2nd layer with 8 neuron that are fully connected
net = tflearn.fully_connected(net, 8)
# Final output layer: allow to get porbability for each output
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try: 
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Hi, I'm the chatbot - How can I help you ? (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
            
        results = model.predict([bag_of_words(inp, words)])
        #get the index of the greatest number to predict the correct respond
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #if the response predicted close to 75% return the predicted message
        if results_index > 0.75:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg['responses']
                    print(random.choice(responses))
        else:
            print("Sorry, I didn't get that Can you say it again?")

chat()