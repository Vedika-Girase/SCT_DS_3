# This script will guide you through building a decision tree classifier
# for the Bank Marketing dataset, as required by your internship task.

# Step 1: Install necessary libraries (if you haven't already)
# You can run this command in your terminal or a Jupyter notebook cell:
# pip install pandas scikit-learn matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

# --- Step 2: Data Loading and Preprocessing ---
# You need to download the 'bank-additional-full.csv' dataset.
# A common way to get it is from the UCI Machine Learning Repository or Kaggle.
# Place the file in the same directory as this script.
print("Loading the dataset...")
try:
    data = pd.read_csv('bank.csv', sep=';')
except FileNotFoundError:
    print("Error: 'bank.csv' not found. Please download it and place it in the same directory.")
    exit()

print("Initial data shape:", data.shape)
print("\nFirst 5 rows of the data:")
print(data.head())

# The target variable 'y' is a categorical value ('yes' or 'no').
# We need to convert it to a numerical format (0 or 1).
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)
print("\nTarget variable 'y' converted to numerical:")
print(data['y'].value_counts())

# The dataset contains several categorical columns that need to be
# converted to a numerical format for the decision tree model.
# This process is called One-Hot Encoding.
categorical_cols = data.select_dtypes(include=['object']).columns.drop('y', errors='ignore')
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print("\nData after one-hot encoding:")
print(data.head())

# --- Step 3: Model Training ---

# Separate the features (X) from the target variable (y).
X = data.drop('y', axis=1)
y = data['y']

# Split the data into training and testing sets.
# We'll use 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Create a Decision Tree Classifier model.
# We'll use a max_depth to prevent the tree from becoming too complex (overfitting).
print("\nTraining the Decision Tree model...")
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model on the training data.
clf.fit(X_train, y_train)
print("Model training complete.")

# --- Step 4: Model Evaluation ---

# Use the trained model to make predictions on the test data.
print("\nEvaluating the model...")
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.4f}")

# Generate a confusion matrix to see where the model made errors.
# This shows how many predictions were correct/incorrect for each class.
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix for better understanding.
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Subscribe', 'Subscribed'],
            yticklabels=['Did Not Subscribe', 'Subscribed'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Step 5: Visualization of the Decision Tree ---

print("\nGenerating a visualization of the Decision Tree...")
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=['No', 'Yes'])
plt.title("Decision Tree Classifier Visualization")
plt.show()
print("Visualization complete. A window with the decision tree will appear.")


  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables)
