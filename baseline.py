import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data
train_df = pd.read_json('data/processed_train.jsonl', lines=True)
dev_df = pd.read_json('data/processed_dev.jsonl', lines=True)

# Separate features and target
X_train = train_df.drop('successful_appeal', axis=1)
y_train = train_df['successful_appeal']
X_dev = dev_df.drop('successful_appeal', axis=1)
y_dev = dev_df['successful_appeal']

# Majority Class Baseline
majority_clf = DummyClassifier(strategy='most_frequent', random_state=42)
majority_clf.fit(X_train, y_train)
y_pred_majority = majority_clf.predict(X_dev)

accuracy = accuracy_score(y_dev, y_pred_majority)
precision = precision_score(y_dev, y_pred_majority)
recall = recall_score(y_dev, y_pred_majority)
f1 = f1_score(y_dev, y_pred_majority)
auc = roc_auc_score(y_dev, y_pred_majority)

print(f"\nMajority Class Baseline Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")