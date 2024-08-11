import numpy as np
import pandas as pd

# Number of samples in the dataset
num_samples = 1000

# Generating synthetic data
data = {
    'Energy_Consumption_kWh': np.random.uniform(1000, 10000, num_samples),
    'Waste_Generated_kg': np.random.uniform(100, 1000, num_samples),
    'Recycling_Rate': np.random.uniform(0, 1, num_samples),
    'Investments_in_Green_Projects_USD': np.random.uniform(10000, 1000000, num_samples),
    'Water_Consumption_Liters': np.random.uniform(1000, 50000, num_samples),
    'Transportation_Emissions_kgCO2e': np.random.uniform(100, 10000, num_samples),
    'Renewable_Energy_Usage_Percentage': np.random.uniform(0, 100, num_samples),
    'Employee_Awareness_Programs_Hours': np.random.uniform(0, 100, num_samples)
}

# Define the relationship for Green Score and Carbon Emissions (simple linear relationships for demonstration)
data['Green_Score'] = (
    0.1 * data['Energy_Consumption_kWh'] +
    0.2 * data['Waste_Generated_kg'] +
    0.3 * data['Recycling_Rate'] +
    0.1 * data['Investments_in_Green_Projects_USD'] +
    0.1 * data['Water_Consumption_Liters'] +
    0.1 * data['Renewable_Energy_Usage_Percentage'] +
    0.1 * data['Employee_Awareness_Programs_Hours'] 
)

data['Carbon_Emissions'] = (
    0.4 * data['Energy_Consumption_kWh'] +
    0.3 * data['Transportation_Emissions_kgCO2e'] +
    0.1 * data['Waste_Generated_kg'] +
    0.1 * data['Water_Consumption_Liters'] +
    0.1 * data['Employee_Awareness_Programs_Hours']
)

# Creating a DataFrame
df = pd.DataFrame(data)

# Save the synthetic dataset to a CSV file
df.to_csv('synthetic_data_extended.csv', index=False)

print("Synthetic dataset created and saved as 'synthetic_data_extended.csv'")
