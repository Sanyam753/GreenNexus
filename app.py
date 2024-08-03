import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the synthetic dataset
df = pd.read_csv('synthetic_carbon_footprint_data.csv')

# One-hot encode the categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['Category', 'Material']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Category', 'Material']))
df_encoded = pd.concat([df, encoded_df], axis=1).drop(['Category', 'Material'], axis=1)

# Define the features (X) and the target variable (y)
X = df_encoded.drop('CarbonEmission', axis=1)
y = df_encoded['CarbonEmission']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Define specific products for each industry
products = {
    "Clothing": [
        {"name": "Cotton Shirt", "material": "Cotton", "image": "images/cotton_shirt.jpg"},
        {"name": "Polyester Jacket", "material": "Polyester", "image": "images/polyester_jacket.jpg"}
    ],
    "Automobile": [
        {"name": "Diesel Car", "material": "Diesel", "image": "images/diesel_car.jpeg"},
        {"name": "Electric Car", "material": "Electric", "image": "images/electric_car.jpg"}
    ],
    "Food": [
        {"name": "Organic Vegetables", "material": "Vegetables", "image": "images/organic_vegetables.jpg"},
        {"name": "Dairy Milk", "material": "Dairy", "image": "images/dairy_milk.jpeg"}
    ],
    "Electronics": [
        {"name": "LED Television", "material": "Television", "image": "images/led_television.jpg"},
        {"name": "Smartphone", "material": "Mobile", "image": "images/smartphone.jpg"}
    ]
}

# Streamlit application
st.title('Carbon Emission Calculator')

# Select industry
industry = st.selectbox("Select Industry", list(products.keys()))

# Display products as images
selected_product = None
cols = st.columns(2)
for i, product in enumerate(products[industry]):
    with cols[i % 2]:
        st.image(product["image"], use_column_width=True)
        if st.button(product["name"]):
            selected_product = product

if selected_product:
    st.subheader(f"Selected Product: {selected_product['name']}")
    new_data = pd.DataFrame({
        'Category': [industry],
        'Material': [selected_product['material']]
    })
    new_data_encoded = encoder.transform(new_data)
    new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoder.get_feature_names_out(['Category', 'Material']))
    predicted_emission = model.predict(new_data_encoded_df)
    st.write(f"Predicted Carbon Emission: {predicted_emission[0]:.2f} units")

    # Display model evaluation metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Model Mean Squared Error: {mse:.2f}")
    st.write(f"Model Mean Absolute Error: {mae:.2f}")
    st.write(f"Model R² Score: {r2:.2f}")
