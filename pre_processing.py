import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# --------------------------- Hyperparameters ---------------------------

# Random seed
RANDOM_SEED = 42

# Data paths
TRAIN_DATA_PATH = 'data/train.jsonl'
DEV_DATA_PATH = 'data/dev.jsonl'
TEST_DATA_PATH = 'data/test.jsonl'
TRAIN_EMBEDDINGS_PATH = 'data/sembed/train.npy'
DEV_EMBEDDINGS_PATH = 'data/sembed/dev.npy'
TEST_EMBEDDINGS_PATH = 'data/sembed/test.npy'

# Feature configurations
FEATURE_COLUMNS = [
    'title',  # x1
    'petitioner_state',  # x4
    'respondent_state',  # x5
    'petitioner_category',  # x6
    'respondent_category',  # x7
    'issue_area',  # x8
    'year',  # x9
    'argument_day',  # x10 (converted from 'argument_date')
    'court_hearing_length',  # x11
    'utterances_number',  # x12
    'court_hearing'  # x13
]

CATEGORICAL_FEATURES = [
    'petitioner_state', 'respondent_state',
    'petitioner_category', 'respondent_category', 'issue_area'
]

TEXT_FEATURES = ['title']

NUMERICAL_FEATURES = ['year', 'argument_day', 'court_hearing_length', 'utterances_number']

# Text vectorization parameters
TFIDF_MAX_FEATURES = 64
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = 'english'

# Feature selection parameters
SELECTK_VALUE = 64

# Ignore warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(RANDOM_SEED)

# --------------------------- Data Loading ---------------------------

# Read data
train_df = pd.read_json(TRAIN_DATA_PATH, lines=True)
dev_df = pd.read_json(DEV_DATA_PATH, lines=True)
test_df = pd.read_json(TEST_DATA_PATH, lines=True)

train_embeddings = np.load(TRAIN_EMBEDDINGS_PATH)
dev_embeddings = np.load(DEV_EMBEDDINGS_PATH)
test_embeddings = np.load(TEST_EMBEDDINGS_PATH)

# Add embeddings to DataFrame
train_df['court_hearing'] = [embedding.tolist() for embedding in train_embeddings]
dev_df['court_hearing'] = [embedding.tolist() for embedding in dev_embeddings]
test_df['court_hearing'] = [embedding.tolist() for embedding in test_embeddings]

# --------------------------- Data Processing ---------------------------

# Convert 'argument_date' to 'argument_day'
reference_date = datetime(1900, 1, 1)

for df in [train_df, dev_df, test_df]:
    df['argument_date'] = pd.to_datetime(df['argument_date'], errors='coerce')
    df['argument_day'] = (df['argument_date'] - reference_date).dt.days

# Split X and y
X_train = train_df[FEATURE_COLUMNS]
y_train = train_df['successful_appeal']

X_dev = dev_df[FEATURE_COLUMNS]
y_dev = dev_df['successful_appeal']

X_test = test_df[FEATURE_COLUMNS]

# Copy dataframes
X_train = X_train.copy()
X_dev = X_dev.copy()
X_test = X_test.copy()

# --------------------------- Plot Class Distribution ---------------------------

# Plot training set class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train)
plt.title('Training Set Class Distribution')
plt.xlabel('Successful Appeal')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print class proportions
class_counts = y_train.value_counts()
print("\nClass distribution in training set:")
print(class_counts)
print("Proportion of class 1 (successful appeal): {:.2f}%".format(100 * class_counts[1] / len(y_train)))
print("Proportion of class 0 (unsuccessful appeal): {:.2f}%".format(100 * class_counts[0] / len(y_train)))

# --------------------------- Preprocessing ---------------------------

# Handle categorical features
encoder = TargetEncoder(cols=CATEGORICAL_FEATURES)
X_train_cat = encoder.fit_transform(X_train[CATEGORICAL_FEATURES], y_train)
X_dev_cat = encoder.transform(X_dev[CATEGORICAL_FEATURES])
X_test_cat = encoder.transform(X_test[CATEGORICAL_FEATURES])

print("Categorical features encoded.")
print("X_train_cat shape:", X_train_cat.shape)
print("X_dev_cat shape:", X_dev_cat.shape)
print("X_test_cat shape:", X_test_cat.shape)

# Process text features
# Title
vectorizer_title = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                   ngram_range=TFIDF_NGRAM_RANGE,
                                   stop_words=TFIDF_STOP_WORDS)

X_train_title = vectorizer_title.fit_transform(X_train[TEXT_FEATURES[0]].fillna('').astype(str)).toarray()
X_dev_title = vectorizer_title.transform(X_dev[TEXT_FEATURES[0]].fillna('').astype(str)).toarray()
X_test_title = vectorizer_title.transform(X_test[TEXT_FEATURES[0]].fillna('').astype(str)).toarray()

print("Title text features extracted.")
print("X_train_title shape:", X_train_title.shape)
print("X_dev_title shape:", X_dev_title.shape)
print("X_test_title shape:", X_test_title.shape)

# Handle numerical features
# Fill missing values with mean
for col in NUMERICAL_FEATURES:
    X_train[col] = X_train[col].fillna(X_train[col].mean())
    X_dev[col] = X_dev[col].fillna(X_train[col].mean())
    X_test[col] = X_test[col].fillna(X_train[col].mean())

# Standardize numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
X_dev_num = scaler.transform(X_dev[NUMERICAL_FEATURES])
X_test_num = scaler.transform(X_test[NUMERICAL_FEATURES])

print("Numerical features standardized.")
print("X_train_num shape:", X_train_num.shape)
print("X_dev_num shape:", X_dev_num.shape)
print("X_test_num shape:", X_test_num.shape)

# Court_hearing embeddings
X_train_hearing = np.array(X_train['court_hearing'].tolist())
X_dev_hearing = np.array(X_dev['court_hearing'].tolist())
X_test_hearing = np.array(X_test['court_hearing'].tolist())

print("Court hearing embeddings processed.")
print("X_train_hearing shape:", X_train_hearing.shape)
print("X_dev_hearing shape:", X_dev_hearing.shape)
print("X_test_hearing shape:", X_test_hearing.shape)

# Combine features
X_train_processed = np.hstack((X_train_cat, X_train_title, X_train_num, X_train_hearing))
X_dev_processed = np.hstack((X_dev_cat, X_dev_title, X_dev_num, X_dev_hearing))
X_test_processed = np.hstack((X_test_cat, X_test_title, X_test_num, X_test_hearing))

print("All features combined.")
print("X_train_processed shape:", X_train_processed.shape)
print("X_dev_processed shape:", X_dev_processed.shape)
print("X_test_processed shape:", X_test_processed.shape)

# --------------------------- Feature Selection ---------------------------

selector = SelectKBest(f_classif, k=SELECTK_VALUE)
X_train_selected = selector.fit_transform(X_train_processed, y_train)
X_dev_selected = selector.transform(X_dev_processed)
X_test_selected = selector.transform(X_test_processed)

print("Feature selection completed.")
print("X_train_selected shape:", X_train_selected.shape)
print("X_dev_selected shape:", X_dev_selected.shape)
print("X_test_selected shape:", X_test_selected.shape)

# Get feature scores and p-values
feature_scores = selector.scores_
feature_pvalues = selector.pvalues_
selected_features = selector.get_support(indices=True)

# Create feature names
# Categorical features (TargetEncoder)
cat_feature_names = CATEGORICAL_FEATURES  # One feature per column with TargetEncoder

# Text features
title_feature_names = [f'title_{word}' for word in vectorizer_title.get_feature_names_out()]

# Numerical features
num_feature_names = NUMERICAL_FEATURES

# Court hearing embeddings
hearing_dim = X_train_hearing.shape[1]
hearing_feature_names = [f'court_hearing_{i}' for i in range(hearing_dim)]

# Combine feature names
all_feature_names = np.concatenate([cat_feature_names, title_feature_names, num_feature_names, hearing_feature_names])

# Get selected feature names
selected_feature_names = all_feature_names[selected_features]

# --------------------------- Plot Feature Importances ---------------------------

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({
    'Feature': all_feature_names,
    'Score': selector.scores_
})

# Sort by score
feature_importances = feature_importances.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Plot the top 20 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Score', y='Feature', data=feature_importances.head(20), palette='viridis')
plt.title('Top 20 Most Important Features based on ANOVA F-value Scores')
plt.xlabel('F-value Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

# --------------------------- Save Preprocessed Data ---------------------------

np.save('data/X_train_selected.npy', X_train_selected)
np.save('data/y_train.npy', y_train.values)
np.save('data/X_dev_selected.npy', X_dev_selected)
np.save('data/y_dev.npy', y_dev.values)
np.save('data/X_test_selected.npy', X_test_selected)

# Save feature selection scores
np.save('data/feature_scores.npy', feature_scores)
np.save('data/feature_pvalues.npy', feature_pvalues)
np.save('data/selected_features.npy', selected_features)
np.save('data/selected_feature_names.npy', selected_feature_names)

print("Preprocessed data saved to 'data/' directory.")
