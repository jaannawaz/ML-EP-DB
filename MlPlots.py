import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import shap
from sklearn.model_selection import learning_curve
import os

# Create 'save' directory if it does not exist
os.makedirs("save", exist_ok=True)

# Load the input file containing all features
input_file = "Final-Normalized-data-EP-Ass-NonAss.csv"
input_df = pd.read_csv(input_file)

# Load the weighted score results file
weighted_scores_file = "stringent_filtered_genes.csv"
weighted_df = pd.read_csv(weighted_scores_file)

# Merge the filtered genes with the input data to add more context
merged_df = weighted_df.merge(input_df, on="Gene", how="left")

# Save the combined result to a new file for detailed analysis
merged_output_file = "complete_filtered_genes.csv"
merged_df.to_csv(merged_output_file, index=False)
print(f"Detailed filtered genes file saved successfully at '{merged_output_file}'.")

# Adjusting the threshold for Weighted Score to reduce the number of selected genes
threshold = 0.4  # Further relaxed threshold to 0.4
filtered_genes_strict = merged_df[merged_df['Weighted_Score'] > threshold]

# Save the newly filtered genes to a file
high_confidence_output_file = "high_confidence_genes.csv"
filtered_genes_strict.to_csv(high_confidence_output_file, index=False)
print(f"High confidence filtered genes file saved successfully at '{high_confidence_output_file}'. Total genes: {filtered_genes_strict.shape[0]}")

# Visualizing Feature Importance for the Selected Genes
# Plotting the feature importance from the original model
feature_importance = {
    "Phen2Gene_Score_normalized": 0.692994,
    "CADD_Score_normalized": 0.113695,
    "MGI_Phenotpe_normalized": 0.084980,
    "Log_FC_normalized": 0.074518,
    "No_Pathways_normalized": 0.033813
}

features = list(feature_importance.keys())
importance_values = list(feature_importance.values())

plt.figure(figsize=(10, 6))
sns.barplot(x=importance_values, y=features, palette="viridis")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Epilepsy Gene Classification Model")
plt.tight_layout()
plt.savefig("save/feature_importance_plot.png")
plt.show()

# Load the trained model
model_file = "trained_model.pkl"
model = joblib.load(model_file)

# Assuming you have the test data (X_test and y_test)
# Ensure that only the features used during training are included
X_test = merged_df.drop(columns=["Gene", "Weighted_Score", "EP_Asso_normalized"])
y_test = merged_df["EP_Asso_normalized"]

# Get the predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("save/roc_curve.png")
plt.show()

# Plotting the Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.savefig("save/confusion_matrix.png")
plt.show()

# Plotting the Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("save/precision_recall_curve.png")
plt.show()

# Feature Distribution
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=input_df, x=feature, hue='EP_Asso_normalized', kde=True)
    plt.title(f'Distribution of {feature} by EP Association')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"save/{feature}_distribution.png")
    plt.show()

# Plotting SHAP Values
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Summary plot to show the impact of features on predictions
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("save/shap_summary_plot.png")
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_test, y_test, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("save/learning_curve.png")
plt.show()
