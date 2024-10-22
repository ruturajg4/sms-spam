from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import os
import logging
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize logging configuration
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure required NLTK data files are downloaded only once
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model_path = 'model1.pkl'
vectorizer_path = 'vectorizer.pkl'

# Ensure that model and vectorizer files exist and are loaded correctly
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            logging.debug("Model loaded successfully.")

        with open(vectorizer_path, 'rb') as file:
            tfidf = pickle.load(file)
            logging.debug("TF-IDF vectorizer loaded successfully.")

        # Check if the TF-IDF vectorizer is already fitted
        if not hasattr(tfidf, 'vocabulary_'):
            logging.info("TF-IDF vectorizer is not fitted. Fitting with a default corpus.")
            default_corpus = ['This is a default message.', 'Spam messages are bad.', 'Ham messages are good.']
            tfidf.fit(default_corpus)
            logging.info("TF-IDF vectorizer has been fitted with the default corpus.")
    except Exception as e:
        logging.error(f"Failed to load model or vectorizer: {str(e)}")
        raise
else:
    logging.error("Model or vectorizer file not found.")
    raise FileNotFoundError("Model or vectorizer file not found.")

# Initialize the Porter Stemmer
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input message from the form
        input_sms = request.form.get('message', '').strip()

        if input_sms:
            logging.debug(f"Received message: {input_sms}")
            # 1. Preprocess the input message
            transformed_sms = transform_text(input_sms)
            logging.debug(f"Transformed message: {transformed_sms}")
            # 2. Vectorize the transformed message
            vector_input = tfidf.transform([transformed_sms])
            logging.debug(f"Vectorized input: {vector_input.toarray()}")  # Convert to array for better logging
            # 3. Predict using the pre-trained model
            result = model.predict(vector_input)[0]
            logging.debug(f"Prediction result: {result}")
            # 4. Display the result
            output = 'Spam' if result == 1 else 'Not Spam'

            return render_template('index.html', prediction_text='Prediction: {}'.format(output))
        else:
            return render_template('index.html', prediction_text='Please enter a message to classify.')
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())  # Log full traceback for debugging
        return render_template('index.html', prediction_text='An error occurred. Please try again.')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
  
