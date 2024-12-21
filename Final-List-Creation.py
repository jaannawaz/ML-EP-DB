import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import joblib

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
threshold = 0.9
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
sns.barplot(x=importance_values, y=features, palette="viridis", hue=features, dodge=False, legend=False)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Epilepsy Gene Classification Model")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.show()

# Plotting the ROC Curve (if ROC data is available from model training)
# Load the trained model
model_file = "trained_model.pkl"
model = joblib.load(model_file)

# Assuming you have the test data (X_test and y_test)
# Exclude any non-feature columns like "Gene", "Weighted_Score", "Association_Type"
X_test = merged_df.drop(columns=["Gene", "Weighted_Score", "Association_Type", "EP_Asso_normalized"])
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
plt.savefig("roc_curve.png")
plt.show()
