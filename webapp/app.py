# webapp/app.py
import streamlit as st
import pandas as pd
import torch
import sys
from pathlib import Path
import joblib
import string
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import json
import spacy
from collections import Counter
from utils.constants import CATEGORY_MAPPING  # import category mapping


# Add src to the path so we can import preprocess_text
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from preprocess import preprocess_text  # Import the preprocess_text function


# function to load NLP models and sentiment analyzer
@st.cache_resource
def get_nlp_models():
    try:
        nltk.data.find("vader_lexicon") # check if vader_lexicon is already downloaded
    except LookupError:
        nltk.download("vader_lexicon", quiet=True) # download if not found

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Load the small English model from spaCy
    analyzer = SentimentIntensityAnalyzer() # Initialize VADER sentiment analyzer

    return nlp, analyzer

# function to load the trained XGBoost model
@st.cache_resource
def get_xgb_model():
    # 1. Create a Path object to your saved model (relative to repo root)
    model_path = Path(__file__).resolve().parents[1] / "scripts" / "model" / "review_classifier.pkl"

    # 2. Load the model using joblib
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return {}

    try:
        best_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return {}

    return {"best_model": best_model}