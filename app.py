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
        {"name": "Polyester Jacket", "material": "Polyester", "image": "images/polyester_jacket.jpg"},
        {"name": "Wool Sweater", "material": "Wool", "image": "images/wool_sweater.jpeg"},
        {"name": "Silk Scarf", "material": "Silk", "image": "images/silk_scarf.jpeg"},
        {"name": "Denim Jeans", "material": "Denim", "image": "images/wool_sweater.jpeg"},
        {"name": "Leather Jacket", "material": "Leather", "image": "images/wool_sweater.jpeg"},
        {"name": "Linen Shirt", "material": "Linen", "image": "images/wool_sweater.jpeg"}
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

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.subheader("Select Options")

# Sidebar options
sidebar_options = ["Home", "Model Evaluation Metrics", "About"]
selected_option = st.sidebar.selectbox("Go to", sidebar_options)

# Sidebar content
if selected_option == "Home":
    st.sidebar.write("Welcome to the Carbon Emission Calculator. Please select an industry and a product to calculate its carbon emission.")
elif selected_option == "Model Evaluation Metrics":
    st.sidebar.write("Model Evaluation Metrics provide insights into the performance of the carbon emission prediction model.")
elif selected_option == "About":
    st.sidebar.write("This application predicts the carbon emission of different products based on their category and material. It uses a Linear Regression model trained on synthetic data.")

# Main page content based on sidebar selection
if selected_option == "Home":
    col1, col2 = st.columns([4, 1])  # Set the ratio of the columns

    with col1:
        # Select industry
        industry = st.selectbox("Select Industry", list(products.keys()))

        # Display products as images with fixed size
        selected_product = None
        cols = st.columns(2)
        for i, product in enumerate(products[industry]):
            with cols[i % 2]:
                st.image(product["image"], width=150)  # Set fixed width
                if st.button(product["name"]):
                    selected_product = product

        if selected_product:
            st.subheader(f"Selected Product: {selected_product['name']}")
            weight = st.slider("Select Weight", 1, 100, 50)
            new_data = pd.DataFrame({
                'Category': [industry],
                'Material': [selected_product['material']],
                'Weight': [weight]  # Add weight here
            })
            new_data_encoded = encoder.transform(new_data[['Category', 'Material']])
            new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoder.get_feature_names_out(['Category', 'Material']))
            new_data_encoded_df['Weight'] = new_data['Weight'].values  # Add weight to the encoded data

            # Ensure the columns are in the same order as X_train
            new_data_encoded_df = new_data_encoded_df[X_train.columns]

            predicted_emission = model.predict(new_data_encoded_df)
            st.write(f"Predicted Carbon Emission: {predicted_emission[0]:.2f} units")

    with col2:
        # Display carbon emission data on the right
        st.subheader("Carbon Emission Data")
        if selected_product:
            st.write(f"Selected Product: {selected_product['name']}")
            st.write(f"Predicted Carbon Emission: {predicted_emission[0]:.2f} units")
        else:
            st.write("Select a product to see the carbon emission data.")

elif selected_option == "Model Evaluation Metrics":
    col1, col2 = st.columns([3, 1])  # Set the ratio of the columns
    with col1:
        # Display model evaluation metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

elif selected_option == "About":
    st.write("This application is designed to help users calculate the carbon emissions of various products based on their category and material. The model used is a Linear Regression model trained on synthetic data.")

# Define CSS for product display
st.markdown(
    """
    <style>
    .product-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
    }
    .product-card {
        padding: 10px;
        border: 2px solid #ddd;
        border-radius: 10px;
        margin: 10px;
        width: 300px;
        box-shadow: 2px 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        text-align: center;
    }
    .product-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .product-card img {
        border-radius: 10px;
        width: 300px;
        height: 300px;
        object-fit: cover;
    }
    
    .product-card button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)
