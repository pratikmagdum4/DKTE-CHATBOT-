import json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:abc@localhost:4306/dkte_chatbot'
db = SQLAlchemy(app)

class Intent(db.Model):
    __tablename__ = 'intents'
    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(255), nullable=False)
    patterns = db.Column(db.Text, nullable=True)
    responses = db.Column(db.Text, nullable=True)
    context_set = db.Column(db.String(255), nullable=True)

def fetch_intents_from_db():
    with app.app_context():
        intents = {"intents": []}
        intents_query = Intent.query.all()
        for intent in intents_query:
            # Convert comma-separated strings to JSON arrays
            patterns = json.dumps(intent.patterns.split(',')) if intent.patterns else '[]'
            responses = json.dumps(intent.responses.split(',')) if intent.responses else '[]'

            # Now decode the JSON strings
            patterns = json.loads(patterns)
            responses = json.loads(responses)

            intent_data = {
                "tag": intent.tag,
                "patterns": patterns,
                "responses": responses,
                "context_set": intent.context_set,
                "id": intent.id
            }
            intents["intents"].append(intent_data)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, 'static', 'data', 'datafromserver.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(intents, json_file, indent=4)
        print("Successfully fetched")

if __name__ == "__main__":
    fetch_intents_from_db()
