import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import os

# Load the model
model_file = "trained_model.pkl"
model = joblib.load(model_file)

# Load the input file containing all features
input_file = "Final-Normalized-data-EP-Ass-NonAss.csv"
input_df = pd.read_csv(input_file)

# Prepare feature data for visualization
X = input_df.drop(columns=["Gene", "EP_Asso_normalized"])  # Features
y = input_df["EP_Asso_normalized"]

# Directory to save the plots
os.makedirs("save", exist_ok=True)

# Part A: Visualize a few Decision Trees from the Random Forest
if isinstance(model, RandomForestClassifier):
    for i in range(3):  # Plot the first 3 trees
        plt.figure(figsize=(20, 10))
        tree.plot_tree(model.estimators_[i], feature_names=X.columns, filled=True, rounded=True, max_depth=3)
        plt.title(f"Decision Tree {i + 1}")
        plt.savefig(f"save/decision_tree_{i + 1}.png")
        plt.close()

# Part B: Correlation Matrix Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation among Predictors')
plt.tight_layout()
plt.savefig("save/correlation_matrix.png")
plt.close()

# Part C: Distribution of Predicted Probability Scores
y_prob = model.predict_proba(X)[:, 1]
true_class = y == 1
false_class = y == 0

plt.figure(figsize=(10, 6))
sns.histplot(y_prob[true_class], color='blue', kde=True, label='True EP Association', bins=20)
sns.histplot(y_prob[false_class], color='red', kde=True, label='False EP Association', bins=20)
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probability Scores')
plt.legend()
plt.tight_layout()
plt.savefig("save/predicted_probability_distribution.png")
plt.close()

print("All plots saved successfully in the 'save' folder.")
