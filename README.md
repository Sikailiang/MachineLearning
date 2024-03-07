# MachineLearning
CMT316
#Text Classification Project

This project utilizes a Naive Bayes classifier for text data classification. The project consists of two main scripts: data.py for data preprocessing and feature_extraction.py for feature extraction, model training, and evaluation.

Installation of Dependencies:

    pip install -r requirements.txt
Data:
In the data.py script, you need to set the value of the data_dir variable to the path of your data folder, for example:
    
    data_dir = '/path/to/your/data'

Running the Scripts:
Data Preprocessing:
Firstly, run the data.py script for data preprocessing. 
This script will clean the text data, tokenize, remove stopwords, and lemmatize the words, then save the  processed text and labels in JSON format: texts_preprocessed.json,labels.json.

    python data_preprocessing.py
Feature Extraction and Model Training:
Next, execute the feature_extraction.py script for feature extraction, followed by training a Naive Bayes model using the extracted features, and evaluating its performance on the test set.

    python feature_extraction_naivebayes.py
This script will output the accuracy of the model on the test set, a classification report, as well as macro-averaged precision, recall, and F1 score.
