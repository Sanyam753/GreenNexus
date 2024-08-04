import pandas as pd
import numpy as np

# Define the product categories and materials
categories = ['Clothing', 'Automobile', 'Electronics', 'Food']
materials = {
    'Clothing': ['Cotton', 'Polyester', 'Wool', 'Silk', 'Nylon'],
    'Automobile': ['Electric', 'Diesel', 'Petrol', 'Hybrid'],
    'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Television', 'Smartwatch'],
    'Food': ['Vegetables', 'Meat', 'Dairy', 'Grains', 'Processed Foods']
}

# Define average carbon emissions for each material (in hypothetical units)
carbon_emissions = {
    'Cotton': 5,
    'Polyester': 10,
    'Wool': 8,
    'Silk': 6,
    'Nylon': 9,
    'Electric': 3,
    'Diesel': 20,
    'Petrol': 15,
    'Hybrid': 10,
    'Smartphone': 15,
    'Laptop': 25,
    'Tablet': 20,
    'Television': 30,
    'Smartwatch': 10,
    'Vegetables': 2,
    'Meat': 25,
    'Dairy': 10,
    'Grains': 5,
    'Processed Foods': 20
}

# Generate synthetic data
data = []
np.random.seed(42)  # For reproducibility

for category in categories:
    for material in materials[category]:
        for _ in range(30):  # Increase the number of entries per category-material pair
            data.append({
                'Category': category,
                'Material': material,
                'Weight': np.random.randint(1, 100),  # Add random weight between 1 and 100
                'CarbonEmission': carbon_emissions[material] + np.random.normal(0, 1)  # Adding some noise
            })

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the DataFrame to ensure randomness
df = df.sample(frac=1).reset_index(drop=True)

# Display the first few entries
print(df.head())

# Ensure at least 100 entries
print(f"Total entries: {len(df)}")

# Save to CSV
df.to_csv('synthetic_carbon_footprint_data.csv', index=False)
