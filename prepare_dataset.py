import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the system path so you can import your own modules
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Import your own preprocessing and feature extraction functions
from preprocess import preprocess_text
from feature_extraction import extract_features

# -----------------------------
# Step 1: Set up file paths
# -----------------------------
# Path to your raw dataset
data_path = Path(__file__).resolve().parent / "fake-reviews.csv"

# Path where you’ll save the processed dataset
output_path = Path(__file__).resolve().parent / "processed-dataset.csv"

# -----------------------------
# Step 2: Load the raw data
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(data_path)

# -----------------------------
# Step 3: Clean text data
# -----------------------------
print("Cleaning text data...")
df["cleaned_text"] = df["text_"].apply(preprocess_text)  # assumes your column is named "text"

# -----------------------------
# Step 4: Extract features
# -----------------------------
include_pos = True  # you can toggle this if needed

print("Extracting features...")
df = extract_features(df, include_pos)

# -----------------------------
# Step 5: Save the processed dataset
# -----------------------------
print("Saving processed dataset...")
df.to_csv(output_path, index=False)

print(f"✅ Processing complete! Saved to: {output_path}")
