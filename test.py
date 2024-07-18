import json
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from chat import get_response, predict_class
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:abc@localhost:4306/dkte_chatbot'
db = SQLAlchemy(app)

class Feedback(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=False, nullable=False)
    phone = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    sub = db.Column(db.String(120), unique=False, nullable=False)
    msg = db.Column(db.String(120), unique=False, nullable=False)
    date = db.Column(db.String(120), unique=False, nullable=True)

class Intent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(255), nullable=False)
    patterns = db.Column(db.Text, nullable=True)
    responses = db.Column(db.Text, nullable=True)
    context_set = db.Column(db.String(255), nullable=True)

@app.route("/feedBack", methods=["GET", "POST"])
def feedBack():
    if request.method == "POST":
        ''' Add entry to database'''
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
model_file_path = os.path.join(current_dir, 'static', 'models', 'chatbot_model2.h5')
model = tf.keras.models.load_model(model_file_path)

# Create recognizer instance for voice input
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def fetch_intents_from_db():
    intents = []
    intents_query = Intent.query.all()
    for intent in intents_query:
        intent_data = {
            "tag": intent.tag,
            "patterns": json.loads(intent.patterns),
            "responses": json.loads(intent.responses),
            "context_set": intent.context_set
        }
        intents.append(intent_data)
    return intents

intents = fetch_intents_from_db()

@app.get("/")
def index_get():
    return render_template("newdkte.html")

@app.post("/predict")
def predict():
    data = request.get_json()
    message = data.get("message")

    # Check if the message is a voice command
    if message == "voice_input":
        text = convert_voice_to_text()
    else:
        text = message
    
    # Predict response
    ints = predict_class(text, model)
    response = get_response(ints, intents)

    # If no intents are found, provide a default response
    if not response:
        response = "I'm sorry, I didn't understand that."
    
    return jsonify({"answer": response})

def convert_voice_to_text():
    print("Listening for voice input...")
    with microphone as source:
        try:
            audio_data = recognizer.listen(source, timeout=5)  # Adjust timeout as needed
            text = recognizer.recognize_google(audio_data)
            print("Voice input:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            return ""
        except sr.RequestError as e:
            print("Error: Could not request results from Google Speech Recognition service; {0}".format(e))
            return ""

if __name__ == "__main__":
    app.run(debug=True)
