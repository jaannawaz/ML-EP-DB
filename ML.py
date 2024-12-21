import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = r"C:\Users\Administrator\Desktop\Fai-work\final-CADD-LogFC-Non-associated-genes\Final-Normalized-data-EP-Ass-NonAss.csv"
data_df = pd.read_csv(file_path)

# Ensure all data is numeric by encoding categorical variables if any
for column in data_df.columns:
    if data_df[column].dtype == 'object':  # If the column type is object, it's likely a categorical feature
        le = LabelEncoder()
        data_df[column] = le.fit_transform(data_df[column])

# Split the data into features (X) and labels (y)
X = data_df.drop('EP_Asso_normalized', axis=1)  # Features
y = data_df['EP_Asso_normalized']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

print("Model trained successfully")
