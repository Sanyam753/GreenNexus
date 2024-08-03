import pandas as pd
import numpy as np

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

print("Synthetic dataset created successfully.")
