#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader vader_lexicon
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet

# Download SpaCy model
python -m spacy download en_core_web_sm

# Start the Streamlit app
streamlit run BA_App6.py
