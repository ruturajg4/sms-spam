from flask import Flask, request, jsonify, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import numpy as np

# Ensure required NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model_path = 'model1.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    tfidf = pickle.load(file)

ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into words
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        # Only keep alphanumeric tokens
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        # Remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        # Apply stemming
        y.append(ps.stem(i))

    # Join the processed words back into a single string
    return " ".join(y)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input message from the form
    input_sms = request.form['message']

    if input_sms.strip() != "":
        # 1. Preprocess the input message
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the transformed message
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict using the pre-trained model
        result = model.predict(vector_input)[0]
        # 4. Display the result
        output = 'Spam' if result == 1 else 'Not Spam'

        return render_template('index.html', prediction_text='Prediction: {}'.format(output))
    else:
        return render_template('index.html', prediction_text='Please enter a message to classify.')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)