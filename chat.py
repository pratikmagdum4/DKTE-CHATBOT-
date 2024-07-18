import random
import json
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import os

lemmatizer = WordNetLemmatizer()

current_dir1 = os.path.dirname(os.path.abspath(__file__))

word_file_path = os.path.join(current_dir1, 'static', 'data', 'words.pkl')
word_file_path2 = os.path.join(current_dir1, 'static', 'data', 'classes.pkl')

words = pickle.load(open(word_file_path, 'rb'))
classes = pickle.load(open(word_file_path2, 'rb'))

current_dir2 = os.path.dirname(os.path.abspath(__file__))
model_file_path  = os.path.join(current_dir2,'static','models','chatbot_model2.h5')

model = tf.keras.models.load_model(model_file_path)

# Function to clean up a sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words from a sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag : {w}")
    return np.array(bag)

# Function to predict the class of a sentence
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response based on predicted intents
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# intents = json.loads(open('newwithchanges.json').read())

current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the JSON file
json_file_path = os.path.join(current_dir, 'static', 'data', 'newwithchanges2.json')

# Use url_for to get the correct URL for the JSON file
with open(json_file_path) as file:
    intents = json.load(file)
# Create Flask app instance
# app = Flask(__name__)
# @app.route('/')
# def getdata():
    
# # Define Flask route
# @app.route('/chatbot', methods=['POST'])
# def chat():
#     data = request.get_json()
#     user_input = data['message']
#     print("The data is "+data)
#     # Predict the class of the user input
#     ints = predict_class(user_input, model)
    
#     # Get a response based on the predicted intent
#     response = get_response(ints, intents)
    
#     return jsonify({'message': response})

if __name__ == '__main__':
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence  = "do you use credit cards"
        sentence = input("You: ")
        if sentence == "quit":
            break
        ints= predict_class(sentence,model)
        resp = get_response(ints,intents)
        print(resp)
    


