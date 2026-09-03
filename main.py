import warnings
warnings.filterwarnings("ignore", message=".*disable_resource_variables.*")
warnings.filterwarnings("ignore", message=".*curses.*")
import nltk
import numpy as np
from jedi.api import classes
from nltk.stem.lancaster import LancasterStemmer
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)



except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent ["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(word.lower()) for word in words if word not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(list(set(labels)))

    training = []
    output = []
    output_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = list(output_empty)
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)
    training = np.array(training)
    output = np.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output) ,f)

tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,512)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net,512)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir="log")

try:
    model.load("chat.model")
    print("Model loaded successfully!")

except Exception as e:
    print("Existing model could not be loaded.")
    print("Reason:", e)
    print("Creating a new model and training...")

    # Create a completely fresh model/session
    model = tflearn.DNN(net, tensorboard_dir="log")

    model.fit(
        training,
        output,
        n_epoch=1000,
        batch_size=8,
        show_metric=True,
        run_id="chat"
    )

    model.save("chat.model")
    print("Model saved successfully!")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chat():
    print("Start talking with the bot! (type exit to stop..)")
    while True:
        inp = input("You : ")
        if inp.lower() == "exit":
            break
        results = model.predict([bag_of_words(inp,words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg["tag"] == tag:
                resoponses = tg["responses"]
        print(random.choice(resoponses))

chat()




