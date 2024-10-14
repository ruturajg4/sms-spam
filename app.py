from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Ensure required NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model_path = 'model1.pkl'
vectorizer_path = 'vectorizer.pkl'

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(vectorizer_path, 'rb') as file:
        tfidf = pickle.load(file)

except FileNotFoundError as e:
    print("File not found:", e)
    raise SystemExit("Model or vectorizer file not found. Please ensure they are in the correct directory.")

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into words
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric tokens and remove stopwords
    y = [i for i in text if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation]
    
    # Apply stemming and join the processed words back into a single string
    return " ".join([ps.stem(i) for i in y])

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input message from the form
        input_sms = request.form['message']

        if input_sms.strip() != "":
            # Preprocess the input message
            transformed_sms = transform_text(input_sms)
            # Vectorize the transformed message
            vector_input = tfidf.transform([transformed_sms])
            # Predict using the pre-trained model
            result = model.predict(vector_input)[0]
            # Display the result
            output = 'Spam' if result == 1 else 'Not Spam'

            return render_template('index.html', prediction_text='Prediction: {}'.format(output))
        else:
            return render_template('index.html', prediction_text='Please enter a message to classify.')

    except Exception as e:
        print("Error occurred during prediction:", e)  # Log the error
        return render_template('index.html', prediction_text='An error occurred during prediction.')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
