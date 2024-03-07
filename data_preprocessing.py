import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import json

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Function Definitions

def clean_text(text):
    """Remove HTML tags, special characters, numbers, and convert to lowercase."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text


def read_texts(data_dir):
    """Read text files and their labels from the specified directory."""
    texts, labels = [], []
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for fname in os.listdir(category_path):
                if fname.endswith('.txt'):
                    file_path = os.path.join(category_path, fname)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        texts.append(clean_text(file.read()))
                    labels.append(category)
    return texts, labels


# Main Function

if __name__ == "__main__":
    # Specify the data directory
    data_dir = 'C:\\Users\\2024\\Desktop\\bbc'

    # Read and clean text data
    texts, labels = read_texts(data_dir)

    # Tokenization
    texts_tokenized = [word_tokenize(text) for text in texts]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    texts_no_stopwords = [[word for word in tokens if word not in stop_words]
                          for tokens in texts_tokenized]

    # Lemmatization (or stemming)
    lemmatizer = WordNetLemmatizer()
    texts_lemmatized = [[lemmatizer.lemmatize(word) for word in tokens]
                        for tokens in texts_no_stopwords]

    # Rebuild text
    texts_preprocessed = [' '.join(tokens) for tokens in texts_lemmatized]
    # Save preprocessed text and labels
    with open('texts_preprocessed.json', 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)
    with open('labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    # Generate texts_preprocessed in JSON format and save it in the same path as the Python file





