# Import necessary libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os
from flask import Flask ,request ,jsonify

import nltk
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer() #It is used for lemmatization, a process that involves reducing words to their base or root form (called lemmas).


# Load intents from the JSON file
# intents = json.loads(open('/content/qa_file.json').read())
# intents = json.loads(open('DKTE_INFO.json').read())
# intents = json.loads(open('newwithchanges.json').read())
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the JSON file
json_file_path = os.path.join(current_dir, 'static', 'data', 'newwithchanges2.json')

# Use url_for to get the correct URL for the JSON file
with open(json_file_path) as file:
    intents = json.load(file)
# Initialize lists to store words, classes, and training documents
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Loop through each intent and pattern to tokenize words and create training data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean up the words
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))  # Remove duplicates and sort

classes = sorted(set(classes))  # Sort classes

# # Save words and classes to pickle files
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))
# Define the path to the data folder within the static folder
data_folder_path = 'static/data/'

# Create the directory if it doesn't exist
os.makedirs(data_folder_path, exist_ok=True)

# Save words and classes to pickle files in the data folder
pickle.dump(words, open(data_folder_path + 'words.pkl', 'wb'))
pickle.dump(classes, open(data_folder_path + 'classes.pkl', 'wb'))
# Create training data
training = []
outputEmpty = [0] * len(classes)

# Loop through each document to create a bag of words and corresponding output row
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle the training data
random.shuffle(training)
training = np.array(training)

# Split the training data into input (X) and output (Y)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build the neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))



# Compile the model with SGD optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=250, batch_size=5, verbose=1)
models_folder_path = os.path.join(current_dir, 'static', 'models')

# Save the trained model
# model.save('chatbot_model2.h5', hist)
model.save(os.path.join(models_folder_path, 'chatbot_model2.h5'), hist)

# print('Done')
# model.save('static/data/chatbot_model.h5', hist)
print('Done')

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
                    print(f"found in bag: {w}")
    return np.array(bag)

# Function to predict the class of a sentence
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.15
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response based on predicted intents
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that."
        
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Predict the class of the user input
    ints = predict_class(user_input, model)

    # Get a response based on the predicted intent
    response = get_response(ints, intents)


    print("ChatBot:", response)


@app.route('/chatbot',methods=['POST'])
def chat():
    
    data = request.get_json()
    user_input = data['message']
    console.log("The data is "+data)
    #Predict the class of the user input
    ints = predict_class(user_input,model)
    #GEt  a response based on the predicted intent
    response = get_response(ints,intents)
    
    return jsonify({'message':response})

if __name__ == '__main__':
    app.run(debug =True)