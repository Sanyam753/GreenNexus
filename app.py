import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the synthetic dataset
df = pd.read_csv('synthetic_carbon_footprint_data.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# One-hot encode the categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Category', 'Material']])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Category', 'Material']))

# Concatenate the encoded features with the original DataFrame
df_encoded = pd.concat([df, encoded_df], axis=1).drop(['Category', 'Material'], axis=1)

# Display the encoded DataFrame
print("First few rows of the encoded DataFrame:")
print(df_encoded.head())

# Define the features (X) and the target variable (y)
X = df_encoded.drop('CarbonEmission', axis=1)
y = df_encoded['CarbonEmission']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model training completed.")

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Example of new data for prediction
new_data = pd.DataFrame({
    'Category': ['Clothing'],
    'Material': ['Cotton']
})

# One-hot encode the new data
new_data_encoded = encoder.transform(new_data)

# Convert to DataFrame
new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoder.get_feature_names_out(['Category', 'Material']))

# Predict the carbon emission
predicted_emission = model.predict(new_data_encoded_df)
print(f"Predicted Carbon Emission: {predicted_emission[0]}")
