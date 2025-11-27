from sklearn.model_selection import GridSearchCV # For hyperparameter tuning
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix # For model evaluation
from sklearn.feature_selection import SelectFromModel # For feature selection
from xgboost import plot_importance # To plot feature importance
from matplotlib import pyplot as plt # To manipulate graphics of data
import numpy as np # For numerical operations
import pandas as pd # To manipulate data
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from xgboost import XGBClassifier # The XGBoost model
from pathlib import Path # To handle file paths
import joblib # To save and load models
import json # To save model metrics

# Load the processed dataset
df = pd.read_csv("processed-dataset.csv") # Load the dataset



# Load the trained baseline model
model_dir = Path("./model") # Directory where the model is saved
baseline_model_path = model_dir / "review_classifier.pkl" # Path to the saved model
model = joblib.load(baseline_model_path) # Load the model
print("Baseline model loaded.") # Confirm model loading


# Prepare features and labels

# Assign features to X
X = pd.get_dummies(
    df.drop(columns=["label", "text_", "cleaned_text"]), # Prepare features (X) by removing text columns and the label column
    columns=["category"] # One-hot encode the 'category' column (convert categorical variable into dummy/indicator variables... so into 1s and 0s
)

# Convert labels to numbers: OR (Original/real Reviews) = 1, CG (Computer-Generated) = 0
label_map = {"OR": 1, "CG": 0}
y = df["label"].map(label_map)

# create train/test split (use stratify to preserve class balance)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41, stratify=y # stratify=y to maintain class distribution which is important for imbalanced datasets
)



# Chooses which features to keep based on feature importance from the baseline model

importances = model.feature_importances_ # Get feature importances from the model

feature_importance_df = pd.DataFrame({ # Create a DataFrame to display feature importances
    'feature': X.columns, # Feature names
    'importance': importances # Corresponding importance scores
}).sort_values('importance', ascending=False) # Sort by importance

print(feature_importance_df) # Print the feature importance DataFrame
plot_importance(model) # Plot feature importance using XGBoost's built-in function
plt.tight_layout() # Adjust layout to prevent overlap
plt.show() # Show the plot 




# Start hyperparameter tuning

# Parameter grid for hyperparameter tuning, it will adjust the model to find the best parameters
parameters = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2.0]
}

# The model to be used in GridSearchCV
xgb_model = XGBClassifier( 
    use_label_encoder=False, # To avoid warning about label encoding
    eval_metric="logloss", # Evaluation metric for the model
    random_state=41, # For reproducibility
)


# Now we will perform grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=xgb_model, # The model to be tuned
    param_grid=parameters, # The parameter grid to search
    scoring="roc_auc", # Evaluation metric to optimize
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=1, # Verbosity, this will update us on the progress
    return_train_score=True, # To return training scores as well
)

grid_search.fit(X_train, y_train)


# Evaluating Best Model

# Get the best parameters and the best model from the grid search
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

# Print the results of the grid search

print("\n" + "=" * 50) # Print a separator line
print("GRID SEARCH RESULTS") 
print("=" * 50) 
print(f"Best parameters: {best_params}") # Print the best hyperparameters found
print(f"Best cross-validation AUC score: {best_score:.4f}") # Print the best AUC score from cross-validation
print("=" * 50)


# ---------------------------
# 3.1 Select important features
# ---------------------------
thresh = 0.020  # Only keep features with importance >= 0.020
selection = SelectFromModel(model, threshold=thresh, prefit=True)

# Transform train/test feature matrices
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)

# Optionally show which features were kept
selected_features = X.columns[selection.get_support()]
print(f"Selected {len(selected_features)} features (threshold={thresh}):")
print(selected_features.tolist())

# Retrain a new XGBClassifier using the best hyperparameters found by grid search
# best_params is defined above from grid_search.best_params_
xgb_selected = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=41)
xgb_selected.fit(select_X_train, y_train)

# Evaluate the selected-features model on the test set
sel_pred = xgb_selected.predict(select_X_test)
sel_prob = xgb_selected.predict_proba(select_X_test)[:, 1]

print("\nSelected-Feature Model Performance on Test Set:")
print("Classification Report:\n", classification_report(y_test, sel_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, sel_pred))
print(f"Test AUC Score (selected features): {roc_auc_score(y_test, sel_prob):.4f}")

best_pred = best_model.predict(X_test) # Make predictions on the test set using the best model
best_prob = best_model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class

print("\nBest Model Performance on Test Set:")
print("Classification Report:\n", classification_report(y_test, best_pred)) # Print classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, best_pred)) # Print confusion matrix
print(f"Test AUC Score: {roc_auc_score(y_test, best_prob):.4f}") # Print AUC score on the test set



# Compare with baseline model

baseline_pred = model.predict(X_test) # Make predictions on the test set using the baseline model
baseline_prob = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class from the baseline model

print("\n" + "=" * 50)
print("BASELINE vs BEST MODEL COMPARISON")
print("=" * 50)
print(f"Baseline AUC: {roc_auc_score(y_test, baseline_prob):.4f}") # Print AUC score of the baseline model
print(f"Best Model AUC: {roc_auc_score(y_test, best_prob):.4f}") # Compare AUC scores

