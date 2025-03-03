from random import uniform

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve
from scipy.stats import uniform, loguniform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib  # For saving and loading models and parameters

from neural_network import PLOT_LEARNING_CURVE

# 设置Seaborn风格
sns.set(style='whitegrid')

# --------------------------- Hyperparameters and Paths ---------------------------

# Control whether to perform hyperparameter search
PERFORM_HYPERPARAMETER_SEARCH = False  # Set to True to perform hyperparameter search
PLOT_LEARNING_CURVE = False

# Path to save the best hyperparameters
BEST_PARAMS_PATH = 'data/best_params_lr.pkl'

# Data paths
X_TRAIN_PATH = 'data/X_train_selected.npy'
Y_TRAIN_PATH = 'data/y_train.npy'
X_DEV_PATH = 'data/X_dev_selected.npy'
Y_DEV_PATH = 'data/y_dev.npy'
X_TEST_PATH = 'data/X_test_selected.npy'

FEATURE_SCORES_PATH = 'data/feature_scores.npy'
FEATURE_PVALUES_PATH = 'data/feature_pvalues.npy'
SELECTED_FEATURES_PATH = 'data/selected_features.npy'
SELECTED_FEATURE_NAMES_PATH = 'data/selected_feature_names.npy'

# Random seed
RANDOM_SEED = 1

# Ignore warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(RANDOM_SEED)

# --------------------------- Load Preprocessed Data ---------------------------

# Load datasets
X_train_selected = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
X_dev_selected = np.load(X_DEV_PATH)
y_dev = np.load(Y_DEV_PATH)
X_test_selected = np.load(X_TEST_PATH)

# Load feature selection details
feature_scores = np.load(FEATURE_SCORES_PATH)
feature_pvalues = np.load(FEATURE_PVALUES_PATH)
selected_features = np.load(SELECTED_FEATURES_PATH)
selected_feature_names = np.load(SELECTED_FEATURE_NAMES_PATH, allow_pickle=True)

# --------------------------- Hyperparameter Search ---------------------------
# Define hyperparameter search space
param_space = {
    'penalty': ['l1', 'l2'],
    'C': loguniform(1e-2, 1e4),
    'solver': ['liblinear'],
    'class_weight': [None],
    'max_iter': [100, 200, 500, 1000]
}

# Initialize model
lr = LogisticRegression(random_state=RANDOM_SEED)

# Perform grid search
random_cv = RandomizedSearchCV(
    estimator=lr,
    param_distributions=param_space,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=3,  # Set verbosity to 0 to reduce output clutter
    random_state=78
)


def search_hyperparameters():
    # Search the best hyperparameters
    print("Performing hyperparameter search...")
    random_cv.fit(X_train_selected, y_train)

    # Get the best hyperparameters
    best_parameters = random_cv.best_params_
    print("\nBest Hyperparameters:")
    print(best_parameters)

    # Save the best hyperparameters to a file
    joblib.dump(best_parameters, BEST_PARAMS_PATH)
    print(f"Best hyperparameters saved to '{BEST_PARAMS_PATH}'.")
    return best_parameters


if PERFORM_HYPERPARAMETER_SEARCH:
    best_params = search_hyperparameters()
else:
    # Try to load the best hyperparameters from file
    try:
        best_params = joblib.load(BEST_PARAMS_PATH)
        print(f"Loaded best hyperparameters from '{BEST_PARAMS_PATH}':")
        print(best_params)
    except FileNotFoundError:
        print(f"No existing best hyperparameters file found at '{BEST_PARAMS_PATH}'.")
        best_params = search_hyperparameters()

# --------------------------- Train Logistic Regression Model ---------------------------

# Initialize the model with the best hyperparameters
model = LogisticRegression(
    penalty=best_params['penalty'],
    C=best_params['C'],
    solver=best_params['solver'],
    max_iter=best_params['max_iter'],
    class_weight=best_params['class_weight'],
    random_state=RANDOM_SEED
)

# Fit the model
model.fit(X_train_selected, y_train)
print("\nModel training completed.")

# --------------------------- Plot Learning Curve ---------------------------

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1):
    train_sizes = np.linspace(0.1, 1.0, 50)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='accuracy', shuffle=True, random_state=RANDOM_SEED
    )
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    train_errors_mean = np.mean(train_errors, axis=1)
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)

    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.2, color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.2, color="g")
    plt.plot(train_sizes, train_errors_mean, '-', color="r", label="Training error")
    plt.plot(train_sizes, test_errors_mean, '-', color="g", label="Validation error")

    plt.title('Learning Curve (Error vs. Training Examples)', fontsize=14)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if PLOT_LEARNING_CURVE:
    plot_learning_curve(model, X_train_selected, y_train, cv=5, n_jobs=-1)

# --------------------------- Evaluate Model on Development Set ---------------------------

# Predict probabilities and labels on development set
y_dev_pred_proba = model.predict_proba(X_dev_selected)[:, 1]
y_dev_pred = model.predict(X_dev_selected)

# Calculate evaluation metrics
accuracy = accuracy_score(y_dev, y_dev_pred)
precision = precision_score(y_dev, y_dev_pred, zero_division=0)
recall = recall_score(y_dev, y_dev_pred)
f1 = f1_score(y_dev, y_dev_pred)
auc = roc_auc_score(y_dev, y_dev_pred_proba)

print(f"\nModel Performance on Development Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# --------------------------- Plot ROC Curve for Development Set ---------------------------

plt.figure(figsize=(8, 6))
plt.plot(*roc_curve(y_dev, y_dev_pred_proba)[:2], label=f'Development Set (AUC = {auc:.4f})', color='b')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Development Set', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------- Evaluate Model on Training Set ---------------------------

# Predict probabilities and labels on training set
y_train_pred_proba = model.predict_proba(X_train_selected)[:, 1]
y_train_pred = model.predict(X_train_selected)

# Calculate evaluation metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, zero_division=0)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_pred_proba)

print(f"\nModel Performance on Training Set:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1-score: {train_f1:.4f}")
print(f"AUC-ROC: {train_auc:.4f}")

# --------------------------- Plot ROC Curve for Training Set ---------------------------

plt.figure(figsize=(8, 6))
plt.plot(*roc_curve(y_train, y_train_pred_proba)[:2], label=f'Training Set (AUC = {train_auc:.4f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Training Set', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------- Feature Importance Visualization ---------------------------

# Get absolute values of coefficients for feature importance
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': np.abs(coefficients)
})

# Sort features by importance
feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), palette='viridis')
plt.title('Top 20 Most Important Features', fontsize=14)
plt.xlabel('Coefficient Magnitude', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.tight_layout()
plt.show()

# --------------------------- Predict on Test Set ---------------------------

# Load test data to get case_ids
test_df = pd.read_json('data/test.jsonl', lines=True)

# Predict labels on test set
y_test_pred = model.predict(X_test_selected)

# Create output DataFrame
output_df = pd.DataFrame({
    'case_id': test_df['case_id'],
    'successful_appeal': y_test_pred
})

# Save predictions to CSV
output_df.to_csv('data/predictions.csv', index=False)
print("\nPredictions saved to 'data/predictions.csv'.")
