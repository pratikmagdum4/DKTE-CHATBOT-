import json
import pickle
import random
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import speech_recognition as sr
from chat import get_response, predict_class
from pymongo import MongoClient
from dotenv import load_dotenv
import os

import nltk
from nltk.stem import WordNetLemmatizer

load_dotenv()  # Load environment variables from .env file

MONGO_URL = os.getenv("MONGO_URL")
app = Flask(__name__)

# MongoDB configuration
app.config['MONGO_URI'] = MONGO_URL
mongo = MongoClient(app.config['MONGO_URI'])
db = mongo.get_database('dkte_chatbot')  # Specify the database name
app.secret_key = '1234567890SECRET'

# Dummy user data
users = {
    'admin': 'password123'
}

# Admin login route
@app.route('/adminLogin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            session['logged_in'] = True
            return redirect(url_for('add_intent'))
        else:
            return 'Invalid credentials. Please try again.'
    return render_template('adminlogin.html')

# Feedback model and route
@app.route("/feedBack", methods=["GET", "POST"])  
def feedBack():
    if request.method == "POST":
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        sub = request.form.get('sub')
        msg = request.form.get('msg')
        feedback_collection = db.feedback
        entry = {
            'name': name,
            'phone': phone,
            'email': email,
            'sub': sub,
            'msg': msg,
            'date': None
        }
        feedback_collection.insert_one(entry)
    return render_template("feedback.html")

# Load intents and model
current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, 'static', 'data', 'datafromserver.json')
with open(json_file_path) as file:
    intents = json.load(file)

model_file_path = os.path.join(current_dir, 'static', 'models', 'chatbot_model_retrained.h5')
model = tf.keras.models.load_model(model_file_path)

# Create recognizer instance for voice input
# recognizer = sr.Recognizer()
# microphone = sr.Microphone()

@app.get("/")
def index_get():
    return render_template("newdkte.html")

lemmatizer = WordNetLemmatizer()
ignoreLetters = ['?', '!', '.', ',']

def retrain_model():
    # Load the updated intents from the JSON file
    with open(json_file_path) as file:
        updated_intents = json.load(file)

    new_words = []
    new_classes = []
    new_documents = []

    for intent in updated_intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            new_words.extend(word_list)
            new_documents.append((word_list, intent['tag']))
            if intent['tag'] not in new_classes:
                new_classes.append(intent['tag'])

    new_words = [lemmatizer.lemmatize(word) for word in new_words if word not in ignoreLetters]
    new_words = sorted(set(new_words))
    new_classes = sorted(set(new_classes))

    data_folder_path = os.path.join(current_dir, 'static', 'data')
    pickle.dump(new_words, open(os.path.join(data_folder_path, 'words.pkl'), 'wb'))
    pickle.dump(new_classes, open(os.path.join(data_folder_path, 'classes.pkl'), 'wb'))

    new_training = []
    output_empty = [0] * len(new_classes)

    for document in new_documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in new_words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[new_classes.index(document[1])] = 1
        new_training.append(bag + output_row)

    random.shuffle(new_training)
    new_training = np.array(new_training)
    train_x = new_training[:, :len(new_words)]
    train_y = new_training[:, len(new_words):]

    new_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    new_model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    models_folder_path = os.path.join(current_dir, 'static', 'models')
    new_model.save(os.path.join(models_folder_path, 'chatbot_model_retrained.h5'))

    print('Model retrained and saved successfully!')

@app.route('/retrain-model', methods=['POST'])
def retrain_route():
    if not session.get('logged_in'):
        return jsonify({'message': 'Unauthorized'}), 401
    retrain_model()
    return jsonify({'message': 'Model retrained successfully!'})

@app.route("/fetch-intents", methods=["GET"])
def fetch_intents():
    intents = {"intents": []}
    intents_query = db.intents.find()
    for intent in intents_query:
        patterns = intent.get('patterns', [])
        responses = intent.get('responses', [])

        intent_data = {
            "tag": intent.get('tag'),
            "patterns": patterns,
            "responses": responses,
            "context_set": intent.get('context_set'),
            "id": str(intent.get('_id'))
        }
        intents["intents"].append(intent_data)
    
    json_file_path = os.path.join(current_dir, 'static', 'data', 'datafromserver.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(intents, json_file, indent=4)
    
    return jsonify(message="Successfully fetched intents")

@app.post("/predict")
def predict():
    data = request.get_json()
    message = data.get("message")
    text = convert_voice_to_text() if message == "voice_input" else message
    ints = predict_class(text, model)
    response = get_response(ints, intents)
    if not response:
        response = "I'm sorry, I didn't understand that."
    return jsonify({"answer": response})

@app.route("/add-intent", methods=["GET","POST"])
def add_intent():
    if not session.get('logged_in'):
        return redirect(url_for('admin_login')) 
    if request.method == "POST":
        tag = request.form['tag']
        patterns = request.form['patterns']
        responses = request.form['responses']
        context_set = request.form.get('context_set', '')
        new_intent = {
            'tag': tag,
            'patterns': patterns.split(','),
            'responses': responses.split(','),
            'context_set': context_set
        }
        db.intents.insert_one(new_intent)
    return render_template("addnewintent.html")

@app.route('/get-feedback')
def get_feedback():
    try:
        feedback_data = list(db.feedback.find())
        return jsonify(feedback_data)
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({"error": "Could not fetch feedback data"}), 500

@app.route('/view-feedback')
def view_feedback():
    feedback_data = list(db.feedback.find())
    return render_template('feedbackDisplay.html', feedback=feedback_data)



if __name__ == "__main__":
    app.run(debug=True)
