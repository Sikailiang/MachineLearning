import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter

# Load preprocessed text and labels
def load_data():
    with open('texts_preprocessed.json', 'r', encoding='utf-8') as f:
        texts_preprocessed = json.load(f)
    with open('labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return texts_preprocessed, labels

# Ensure NLTK resources for POS tagging and tokenization are downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def lexical_diversity(texts):
    """Calculate lexical diversity: ratio of unique vocabulary to total vocabulary."""
    return np.array([len(set(text.split())) / float(len(text.split())) for text in texts]).reshape(-1, 1)

def pos_frequency(texts):
    """Calculate POS tag frequency in texts."""
    pos_counts = []
    for text in texts:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        counts = Counter(tag for word, tag in tagged)
        total_count = sum(counts.values())
        # Take only frequencies of nouns (NN) and verbs (VB)
        pos_counts.append([counts.get('NN', 0) / total_count, counts.get('VB', 0) / total_count])
    return np.array(pos_counts)

# Feature extraction
def extract_features(texts):
    global vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(texts)
    # New features
    lexical_diversities = lexical_diversity(texts)
    pos_freqs = pos_frequency(texts)

    # Convert new features to sparse matrix format and stack with X_tfidf
    lexical_diversity_sparse = csr_matrix(lexical_diversities)
    pos_freq_sparse = csr_matrix(pos_freqs)

    X_combined = hstack([X_tfidf, lexical_diversity_sparse, pos_freq_sparse])
    return X_combined

# Encode labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    return y_encoded

# Split dataset
def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,
                                                      random_state=42)  # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    texts_preprocessed, labels = load_data()
    X_combined = extract_features(texts_preprocessed)
    y_encoded = encode_labels(labels)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_combined, y_encoded)

    # Model training
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred_test = model.predict(X_test)
    print("test set accuracy:", accuracy_score(y_test, y_pred_test))
    print("test set classification report:\n", classification_report(y_test, y_pred_test))
    # Calculate macro-averaged precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
    print(f"macro-averaged precision: {precision}")
    print(f"macro-averaged recall: {recall}")
    print(f"macro-averaged F1: {f1_score}")

    # Check a sample instance in the test set to verify the model output
    test_index = 25
    # Predicted label by the model for this instance
    predicted_label = y_pred_test[test_index]
    print("predicted labels:", predicted_label)

    # True label for this instance
    true_label = y_test[test_index]
    print("true labels:", true_label)

    # Original text
    original_text = texts_preprocessed[test_index]
    print("raw text:", original_text)

    # Check if the prediction is accurate
    if predicted_label == true_label:
        print("The prediction for this instance is accurate.")
    else:
        print("The prediction for this instance is inaccurate.")
