# training.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Define the industries and their respective parameters
industries = {
    "Clothing": {
        "Material": ["Cotton", "Polyester", "Wool", "Silk"],
        "ProductionProcess": ["Handmade", "Machine-made"],
        "TransportDistance": np.random.uniform(50, 1000, 100),
        "EnergyUsed": np.random.uniform(100, 1000, 100)
    },
    "Automobile": {
        "FuelType": ["Diesel", "Petrol", "Electric", "Hybrid"],
        "EngineSize": np.random.uniform(1.0, 4.0, 100),
        "Mileage": np.random.uniform(10, 50, 100),
        "ManufacturingLocation": ["Local", "International"]
    },
    "Electronics": {
        "Type": ["Television", "Mobile", "Laptop", "Tablet"],
        "ProductionEnergy": np.random.uniform(200, 2000, 100),
        "Lifespan": np.random.uniform(1, 10, 100),
        "TransportDistance": np.random.uniform(50, 1000, 100)
    },
    "Food": {
        "Type": ["Grains", "Dairy", "Vegetables", "Fruits"],
        "FarmingMethod": ["Organic", "Conventional"],
        "Processing": ["None", "Minimal", "High"],
        "Packaging": ["Plastic", "Glass", "Cardboard"],
        "TransportDistance": np.random.uniform(50, 1000, 100)
    }
}

# Generate synthetic data for each industry
synthetic_data = []

for industry, params in industries.items():
    for i in range(100):
        row = {"Industry": industry}
        for param, values in params.items():
            if isinstance(values, list):
                row[param] = np.random.choice(values)
            else:
                row[param] = values[i]
        # Generate a random carbon emission value
        row["CarbonEmission"] = np.random.uniform(1, 100)
        synthetic_data.append(row)

# Create a DataFrame
df_synthetic = pd.DataFrame(synthetic_data)

# Save to CSV
df_synthetic.to_csv('synthetic_carbon_footprint_data.csv', index=False)

# Print the columns of the DataFrame to verify
print("DataFrame columns:", df_synthetic.columns)

# Identify categorical and numerical features
categorical_features = ['Industry', 'Material', 'ProductionProcess', 'FuelType', 'ManufacturingLocation', 'Type', 'FarmingMethod', 'Processing', 'Packaging']
numerical_features = [col for col in df_synthetic.columns if col not in categorical_features + ['CarbonEmission']]

# Print to verify
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values for numerical features
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values for categorical features
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X = df_synthetic.drop('CarbonEmission', axis=1)
y = df_synthetic['CarbonEmission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Mean Squared Error: {mse:.2f}")
print(f"Model Mean Absolute Error: {mae:.2f}")
print(f"Model R² Score: {r2:.2f}")

# Save the trained model
joblib.dump(pipeline, 'carbon_emission_model.joblib')
print("Model saved successfully.")
