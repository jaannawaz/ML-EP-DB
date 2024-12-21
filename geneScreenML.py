import pandas as pd
import joblib

# Load the trained model (assuming you saved it as 'trained_model.pkl')
model = joblib.load('trained_rf_model.pkl')

# Load the dataset used for training
file_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\Final-Normalized-data-EP-Ass-NonAss.csv"
data_df = pd.read_csv(file_path)

# Extract features and target
X = data_df.drop(columns=['Gene', 'EP_Asso_normalized'])  # Features
y = data_df['EP_Asso_normalized']  # Target (1 for associated, 0 for non-associated)

# Get the feature importances
importances = model.feature_importances_

# Create a DataFrame for feature importances
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(importance_df)

# Calculate a weighted score for each gene based on feature importance
genes = data_df['Gene']
weighted_scores = X.dot(importances)

# Create a DataFrame to associate genes with their weighted scores
gene_importance_df = pd.DataFrame({'Gene': genes, 'Weighted_Score': weighted_scores})

# Sort genes by their weighted score to determine significance
significant_genes_df = gene_importance_df.sort_values(by='Weighted_Score', ascending=False)

# Display the top significant genes
top_n = 10  # Set this to the number of top genes you want to explore
print(f"\nTop {top_n} Significant Genes Based on Model:")
print(significant_genes_df.head(top_n))

# Optionally, save the significant genes to a CSV file
output_csv_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\significant_genes.csv"
significant_genes_df.to_csv(output_csv_path, index=False)
print(f"\nSignificant genes saved successfully at '{output_csv_path}'")
