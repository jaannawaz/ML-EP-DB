import pandas as pd
import joblib

# Load the trained model
trained_model_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\trained_model.pkl"
model = joblib.load(trained_model_path)

# Load the input data for predictions
input_data_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\Final-Normalized-data-EP-Ass-NonAss.csv"
data_df = pd.read_csv(input_data_path)

# Separate features (excluding Gene column and target column if exists)
X = data_df.drop(columns=['Gene', 'EP_Asso_normalized'])

# Predict probabilities (since we need probabilities to calculate weighted scores)
probs = model.predict_proba(X)[:, 1]  # Probabilities for the positive class (epilepsy association)

# Create a DataFrame with the genes and their weighted scores
gene_scores_df = pd.DataFrame({
    'Gene': data_df['Gene'],
    'Weighted_Score': probs
})

# Classify the genes based on weighted scores
def classify_gene(score):
    if score > 0.5:
        return 'Strongly Associated'
    elif score > 0:
        return 'Weakly Associated'
    else:
        return 'No Association'

gene_scores_df['Association_Type'] = gene_scores_df['Weighted_Score'].apply(classify_gene)

# Save the classified gene data to a CSV file
output_file_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\classified_genes.csv"
gene_scores_df.to_csv(output_file_path, index=False)

print(f"Classified gene data saved successfully to '{output_file_path}'")

# Optional: Display a summary of classified genes
print(gene_scores_df['Association_Type'].value_counts())
