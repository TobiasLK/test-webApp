#!/bin/bash

python -m spacy download en_core_web_sm

python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader vader_lexicon
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4

exec streamlit run BA_App6.py
