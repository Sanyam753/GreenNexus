import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the synthetic dataset
df = pd.read_csv('synthetic_data_extended.csv')

# Features and target variables
X = df[['Energy_Consumption_kWh', 'Waste_Generated_kg', 'Recycling_Rate', 
        'Investments_in_Green_Projects_USD', 'Water_Consumption_Liters', 
        'Transportation_Emissions_kgCO2e', 'Renewable_Energy_Usage_Percentage', 
        'Employee_Awareness_Programs_Hours']]

y_green_score = df['Green_Score']
y_carbon_emissions = df['Carbon_Emissions']

# Split the data into training and testing sets
X_train, X_test, y_train_green, y_test_green, y_train_carbon, y_test_carbon = train_test_split(
    X, y_green_score, y_carbon_emissions, test_size=0.2, random_state=0)

# Train models
model_green = LinearRegression()
model_green.fit(X_train, y_train_green)

model_carbon = LinearRegression()
model_carbon.fit(X_train, y_train_carbon)

# Predict and evaluate
y_pred_green = model_green.predict(X_test)
y_pred_carbon = model_carbon.predict(X_test)

print("Green Score Model RMSE:", mean_squared_error(y_test_green, y_pred_green, squared=False))
print("Carbon Emissions Model RMSE:", mean_squared_error(y_test_carbon, y_pred_carbon, squared=False))

# Save the models
joblib.dump(model_green, 'model_green_extended.pkl')
joblib.dump(model_carbon, 'model_carbon_extended.pkl')

print("Models saved as 'model_green_extended.pkl' and 'model_carbon_extended.pkl'")
