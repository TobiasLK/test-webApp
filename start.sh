#!/bin/bash

python -m spacy download en_core_web_sm

exec streamlit run BA_App6.py
