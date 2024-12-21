import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load the trained RandomForestClassifier model
# Assuming 'model' was previously trained and is available in your script

# Load the CSV file with normalized data
file_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\Final-Normalized-data-EP-Ass-NonAss.csv"
data_df = pd.read_csv(file_path)

# Features and labels
X = data_df.drop(columns=['Gene', 'Class'])  # Features

# Calculate feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Print feature importance
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()
