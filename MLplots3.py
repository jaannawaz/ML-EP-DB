import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
input_file = "Final-Normalized-data-EP-Ass-NonAss.csv"
input_df = pd.read_csv(input_file)

# Print columns to identify the correct column name
print(input_df.columns)

# Update these column names with actual names from your DataFrame
plt.figure(figsize=(12, 8))

# Replace 'Weighted_Score' and 'Prediction_Probability' with correct column names from your dataset
# Example:
x_column = 'CADD_Score_normalized'  # replace this with your chosen feature
y_column = 'Phen2Gene_Score_normalized'  # replace this with your chosen feature or score

# Example: Assuming `confidence_level` column in the dataset defines categories 1, 2, 3, etc.
sns.scatterplot(
    data=input_df,
    x=x_column,
    y=y_column,
    hue='EP_Asso_normalized',  # Assuming 'EP_Asso_normalized' is in your DataFrame to indicate EP association
    palette='viridis',
    s=100  # Marker size
)

plt.axhline(y=0.5, color='black', linestyle='--')  # Adding a horizontal threshold line (if needed)
plt.xlabel(f'{x_column}')
plt.ylabel(f'{y_column}')
plt.title('Scatter Plot of Features in the Dataset')
plt.tight_layout()
plt.savefig("save/scatter_probability_plot.png")
plt.show()
