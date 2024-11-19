# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Loading the data
url = "https://www.kaggleusercontent.com/creditcard.csv"  # Replace with the dataset path
df = pd.read_csv(url)

# View the first few rows of the dataset
print(df.head())

# 2. Understanding the dataset
print(df.info())
print(df.describe())

# Checking class balance
sns.countplot(x='Class', data=df)
plt.title('Distribution of Fraudulent and Normal Transactions')
plt.show()

# 3. Data preparation
# Splitting into independent variables (X) and target variable (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Training the model
clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)

# 5. Evaluating the model
y_pred = clf.predict(X_test)

# Evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# 6. Visualizing Feature Importance
importances = clf.feature_importances_
features = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), features[sorted_indices], rotation=90)
plt.title("Feature Importance")
plt.show()
