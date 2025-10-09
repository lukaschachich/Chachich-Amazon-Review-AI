import pandas as pd
import string
import spacy # Tokenize text so we can count each POS
from collections import Counter # To keep a dictionary of the count of each POS

# ---------------------------
# Feature extraction
# ---------------------------

def extract_features(df, include_pos=False):
    # ensure text is string
    df['cleaned_text'] = df['cleaned_text'].astype(str)

    # Example feature: Length of the review text
    df['char_length'] = df['cleaned_text'].apply(len) # create a new column with the character length of the review text

    df["word_count"] = df["cleaned_text"].str.split().apply(len) # create a new column with the word count of the review text
    
    # Example feature: Count of punctuation marks in the review text
    # vectorized version for speed
    df["punctuation_ct"] = df["cleaned_text"].apply(lambda s: sum(1 for ch in s if ch in string.punctuation)) # count punctuation characters

    df["is_extreme_star"] = df["rating"].isin([1.0, 5.0]) # True if rating is 1 or 5

    # Optionally add POS features
    if include_pos:
        df = add_pos_features(df)

    return df

# ---------------------------
# POS tagging setup
# ---------------------------

nlp = spacy.load("en_core_web_sm", disable=['parser','ner'])  # Load the small English model from spaCy

POS_WHITELIST = {"VERB", "NOUN", "ADV"} # Set of POS tags we want to count

def pos_counts(text): # Function to count POS in a given text
    doc = nlp(text)  # tokenizes the text
    return Counter(
        token.pos_ for token in doc if token.pos_ in POS_WHITELIST
    )

# ---------------------------
# Add POS features to dataframe

def add_pos_features(df): 
    pos_data = df["cleaned_text"].apply(pos_counts)
    pos_df = pd.DataFrame(list(pos_data)).fillna(0)  # fill null counts with 0
    pos_df.index = df.index  # align columns with original df (dataframe)
    return pd.concat([df, pos_df], axis=1)