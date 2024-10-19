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
except pickle.UnpicklingError as e:
    print("Error loading pickle files:", e)
    raise SystemExit("An error occurred while loading the model or vectorizer.")

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    try:
        # Convert text to lowercase
        text = text.lower()
        # Tokenize the text into words
        text = nltk.word_tokenize(text)
        print("Tokenized text:", text)  # Debug log

        # Filter out non-alphanumeric tokens and stopwords
        filtered_words = [i for i in text if i.isalnum() and i not in stopwords.words('english')]
        print("Filtered words (no stopwords, punctuation):", filtered_words)  # Debug log
        
        # Apply stemming
        stemmed_words = [ps.stem(i) for i in filtered_words]
        print("Stemmed words:", stemmed_words)  # Debug log

        # Join the processed words back into a string
        return " ".join(stemmed_words)
    
    except Exception as e:
        print("Error during text transformation:", e)
        raise

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
            print("Input SMS:", input_sms)  # Debug log

            # Preprocess the input message
            transformed_sms = transform_text(input_sms)
            print("Transformed SMS:", transformed_sms)  # Debug log

            # Vectorize the transformed message
            vector_input = tfidf.transform([transformed_sms])
            print("Vector input shape:", vector_input.shape)  # Debug log

            # Predict using the pre-trained model
            result = model.predict(vector_input)[0]
            print("Model prediction result:", result)  # Debug log

            # Display the result
            output = 'Spam' if result == 1 else 'Not Spam'

            return render_template('index.html', prediction_text='Prediction: {}'.format(output))
        else:
            return render_template('index.html', prediction_text='Please enter a message to classify.')

    except FileNotFoundError as e:
        print("File not found error:", e)
        return render_template('index.html', prediction_text="File not found: model or vectorizer.")
    except ValueError as e:
        print("Value error during prediction:", e)
        return render_template('index.html', prediction_text="Value error: Invalid input for prediction.")
    except Exception as e:
        print("Error occurred during prediction:", e)  # Log the error
        return render_template('index.html', prediction_text='An unexpected error occurred during prediction.')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
