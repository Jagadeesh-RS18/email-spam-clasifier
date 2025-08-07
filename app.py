from flask import Flask, render_template, request
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Routes    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form['email_text']
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)

        result = "Spam ❌" if prediction[0] == 1 else "Not Spam ✅"

        return render_template('result.html', email=email_text, result=result)
    except Exception as e:
        return f"Error: {e}"

# Run
if __name__ == '__main__':
    app.run(debug=True)
