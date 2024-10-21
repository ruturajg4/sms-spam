from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

# Ensure required NLTK data files are downloaded only once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model_path = 'model1.pkl'
vectorizer_path = 'vectorizer.pkl'

# Ensure that model and vectorizer files exist and are loaded correctly
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(vectorizer_path, 'rb') as file:
        tfidf = pickle.load(file)
else:
    raise FileNotFoundError("Model or vectorizer file not found.")

ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters, stopwords, and punctuation
    filtered_words = [
        ps.stem(word) for word in words 
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation
    ]

    # Join the processed words back into a single string
    return " ".join(filtered_words)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input message from the form
    input_sms = request.form.get('message', '').strip()

    if input_sms:
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
