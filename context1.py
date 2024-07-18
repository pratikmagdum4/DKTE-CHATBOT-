import json
import pickle
import random
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import speech_recognition as sr
from flask_sqlalchemy import SQLAlchemy
import os
import nltk
from nltk.stem import WordNetLemmatizer
import mysql.connector
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:abc@localhost:4306/dkte_chatbot'
db = SQLAlchemy(app)
app.secret_key = '1234567890SECRET'

# Dummy user data
users = {
    'admin': 'password123'
}

# LangChain LLM and Memory setup
def get_llm():
    model_kwargs = {
        "maxTokens": 1024,
        "temperature": 0.9,
        "topP": 0.5,
        "stopSequences": ["Human:"],
        "countPenalty": {"scale": 0},
        "presencePenalty": {"scale": 0},
        "frequencyPenalty": {"scale": 0}
    }
    llm = Bedrock(
        credentials_profile_name='default',
        model_id="ai21.j2-ultra-v1",
        model_kwargs=model_kwargs
    )
    return llm

def get_memory():
    llm = get_llm()
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=512)
    return memory

# Initialize memory
memory = get_memory()

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
class Feedback(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=False, nullable=False)
    phone = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    sub = db.Column(db.String(120), unique=False, nullable=False)
    msg = db.Column(db.String(120), unique=False, nullable=False)
    date = db.Column(db.String(120), unique=False, nullable=True)

@app.route("/feedBack", methods=["GET", "POST"])
def feedBack():
    if request.method == "POST":
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        sub = request.form.get('sub')
        msg = request.form.get('msg')
        entry = Feedback(name=name, email=email, phone=phone, sub=sub, msg=msg)
        db.session.add(entry)
        db.session.commit()
    return render_template("feedback.html")

# Load intents and model
current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, 'static', 'data', 'datafromserver.json')
with open(json_file_path) as file:
    intents = json.load(file)

model_file_path = os.path.join(current_dir, 'static', 'models', 'chatbot_model_retrained.h5')
model = tf.keras.models.load_model(model_file_path)

# Create recognizer instance for voice input
recognizer = sr.Recognizer()
microphone = sr.Microphone()

@app.get("/")
def index_get():
    return render_template("newdkte.html")

lemmatizer = WordNetLemmatizer()
ignoreLetters = ['?', '!', '.', ',']

def retrain_model():
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
    intents_query = Intent.query.all()
    for intent in intents_query:
        patterns = intent.patterns.split(',') if intent.patterns else []
        responses = intent.responses.split(',') if intent.responses else []

        intent_data = {
            "tag": intent.tag,
            "patterns": patterns,
            "responses": responses,
            "context_set": intent.context_set,
            "id": intent.id
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

    chat_response = get_chat_response(text, memory)  # Get response using memory

    return jsonify({"answer": chat_response})

def get_chat_response(input_text, memory):
    llm = get_llm()
    conversation_with_summary = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    chat_response = conversation_with_summary.predict(input=input_text)
    return chat_response

class Intent(db.Model):
    __tablename__ = 'intents'
    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(255), nullable=False)
    patterns = db.Column(db.Text, default=None)
    responses = db.Column(db.Text, default=None)
    context_set = db.Column(db.String(255), default=None)

@app.route("/add-intent", methods=["GET","POST"])
def add_intent():
    if not session.get('logged_in'):
        return redirect(url_for('admin_login')) 
    if request.method == "POST":
        tag = request.form['tag']
        patterns = request.form['patterns']
        responses = request.form['responses']
        context_set = request.form.get('context_set', '')
        new_intent = Intent(tag=tag, patterns=patterns, responses=responses, context_set=context_set)
        db.session.add(new_intent)
        db.session.commit()
    return render_template("addnewintent.html")

def convert_voice_to_text():
    print("Listening for voice input...")
    with microphone as source:
        try:
            audio_data = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio_data)
            print("Voice input:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"Error: Could not request results from Google Speech Recognition service; {e}")
            return ""

@app.route('/get-feedback')
def get_feedback():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='abc',
            database='dkte_chatbot',
            port=4306
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM feedback")
        feedback_data = cursor.fetchall()
        cursor.close()
        connection.close()
        return jsonify(feedback_data)
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({"error": "Could not fetch feedback data"}), 500
    return render_template ("feedbackDisplay.html")

@app.route('/view-feedback')
def view_feedback():
    feedback_data = Feedback.query.all()
    return render_template('feedbackDisplay.html', feedback=feedback_data)

if __name__ == "__main__":
    app.run(debug=True)
