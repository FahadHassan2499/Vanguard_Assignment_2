import pandas as pd
from sklearn.model_selection import train_test_split
import nltk  # For text preprocessing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True) # For tokenization

def load_and_preprocess_data(data_path="imdb_reviews.csv"): # Replace with your actual path
    try:
        df = pd.read_csv(data_path)  # Load IMDB dataset
        # Preprocessing
        df['review'] = df['review'].apply(preprocess_text)
        X = df['review']
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during data loading: {e}")

def preprocess_text(text):
    # 1. Lowercasing
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # 3. Remove non-alphanumeric characters and keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # 4. Tokenization
    words = nltk.word_tokenize(text)
    # 5. Stop word removal
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # 6. Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # 7. Join words back into a string
    return " ".join(words)


# Example usage (in your training script)
X_train, X_test, y_train, y_test = load_and_preprocess_data()

