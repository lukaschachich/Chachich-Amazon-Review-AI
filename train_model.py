import pandas as pd # To manipulate data
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix # To evaluate model performance
from xgboost import XGBClassifier # The XGBoost model
from pathlib import Path # To handle file paths
import joblib # To save and load models
import json # To save model metrics

df = pd.read_csv("processed-dataset.csv") # Load the dataset

# Check to see if the data loaded correctly
print(f"Dataset loaded with {len(df)} reviews") 
print(f"Columns in dataset: {list(df.columns)}")


# Prepare features and labels

# X variables will be all features
X = pd.get_dummies(
    df.drop(columns=["label", "text_", "cleaned_text"]), # Prepare features (X) by removing text columns and the label column
    columns=["category"] # One-hot encode the 'category' column (convert categorical variable into dummy/indicator variables... so into 1s and 0s
)

# Convert labels to numbers: OR (Original/real Reviews) = 1, CG (Computer-Generated) = 0

label_map = {"OR": 1, "CG": 0} 
y = df["label"].map(label_map)

print(f"Features shape: {X.shape}") # Check the shape of the features, should be (num_reviews, num_features)

print("Labels distribution:")
print(y.value_counts()) # Check the distribution of labels, to check if CG and OR reviews are balanced




# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=41) # random_state is basically like a seed for the random number generator, so you get the same split every time

print(X_train.shape, X_test.shape) # Check the shape of the training and testing sets



# Create XGBoost classifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=41
)
print("Training the model...")

# Train the model
model.fit(X_train, y_train)
print("Model training completed!")



# Make predictions
# Make predictions on the test set

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Print detailed classification report

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print AUC score
auc_score = roc_auc_score(y_test, y_prob)
print(f"\nAUC Score: {auc_score:.4f}")





# Create directory for saving the model

model_dir = Path("./model") # Directory to save the model
model_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

# Save the trained model

joblib.dump(model, model_dir / "review_classifier.pkl") # Save the model using joblib

# Save feature names for future use

feature_names = X.columns.tolist()

with open(model_dir / "feature_names.json", "w") as f:
    json.dump(feature_names, f)

# Save model metadata

model_metadata = {
    "test_auc_score": float(auc_score),
    "num_features": len(feature_names),
    "label_mapping": label_map,
    "training_samples": len(X_train),
    "test_samples": len(X_test)

}

with open(model_dir / "model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

print(f"\nModel saved successfully in '{model_dir}' directory!")