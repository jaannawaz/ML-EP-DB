import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import joblib  # for saving the model

# Load the CSV file with normalized data
file_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\Final-Normalized-data-EP-Ass-NonAss.csv"
data_df = pd.read_csv(file_path)

# Features and labels
X = data_df.drop(columns=['Gene', 'EP_Asso_normalized'])  # Features
y = data_df['EP_Asso_normalized']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\trained_rf_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved successfully at '{model_path}'")

# Predict probabilities for the test set and calculate ROC-AUC score
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Model trained successfully. ROC-AUC Score: {roc_auc:.4f}")

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

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
