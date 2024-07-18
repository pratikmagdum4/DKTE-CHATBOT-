import json
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from chat import get_response, predict_class
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

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
json_file_path = os.path.join(current_dir, 'static', 'data', 'newwithchanges2.json')
with open(json_file_path) as file:
    intents = json.load(file)

model_file_path = os.path.join(current_dir, 'static', 'models', 'chatbot_model2.h5')
model = tf.keras.models.load_model(model_file_path)

# Create recognizer instance for voice input
recognizer = sr.Recognizer()
microphone = sr.Microphone()

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
    print("hi")
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
