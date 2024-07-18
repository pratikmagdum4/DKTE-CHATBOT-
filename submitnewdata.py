from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///intents.db'
db = SQLAlchemy(app)

class Intent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(255), nullable=False)
    patterns = db.Column(db.Text, default=None)
    responses = db.Column(db.Text, default=None)
    context_set = db.Column(db.String(255), default=None)

db.create_all()
@app.route('/submit-intent', methods=['POST'])
def submit_intent():
    tag = request.form['tag']
    patterns = request.form['patterns']
    responses = request.form['responses']
    context_set = request.form.get('context_set', '')  # Optional field

    new_intent = Intent(tag=tag, patterns=patterns, responses=responses, context_set=context_set)
    db.session.add(new_intent)
    db.session.commit()


    return jsonify(message="Intent added successfully"), 201


if __name__ == '__main__':
    app.run(debug=True)
