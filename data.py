import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
        {"name": "Polyester Jacket", "material": "Polyester", "image": "images/polyester_jacket.webp"},
        {"name": "Wool Sweater", "material": "Wool", "image": "images/wool_sweater.jpeg"},
        {"name": "Silk Scarf", "material": "Silk", "image": "images/silk_scarf.jpeg"},
        {"name": "Denim Jeans", "material": "Denim", "image": "images/denim_jeans.jpg"},
        {"name": "Leather Jacket", "material": "Leather", "image": "images/leather_jacket.jpg"},
        {"name": "Linen Shirt", "material": "Linen", "image": "images/linen_shirt.webp"},
        {"name": "Nylon Shorts", "material": "Nylon", "image": "images/nylon_shorts.jpg"}
    ],
    "Automobile": [
        {"name": "Diesel Car", "material": "Diesel", "image": "images/diesel_car.jpeg"},
        {"name": "Electric Car", "material": "Electric", "image": "images/electric_car.jpg"},
        {"name": "Petrol Car", "material": "Petrol", "image": "images/petrol_car.webp"},
        {"name": "Hybrid Car", "material": "Hybrid", "image": "images/hybrid_car.webp"},
        {"name": "SUV", "material": "SUV", "image": "images/suv.jpg"},
        {"name": "Sedan", "material": "Sedan", "image": "images/sedan.jpg"}
    ],
    "Food": [
        {"name": "Organic Vegetables", "material": "Vegetables", "image": "images/organic_vegetables.jpg"},
        {"name": "Dairy Milk", "material": "Dairy", "image": "images/dairy_milk.jpeg"}
    ],
    "Electronics": [
        {"name": "LED Television", "material": "Television", "image": "images/led_television.jpg"},
        {"name": "Smartphone", "material": "Tablet", "image": "images/smartphone.jpg"}
    ]
}

# Streamlit application
st.title('Carbon Emission Data Analytics')

# Add custom CSS to adjust padding
st.markdown("""
    <style>
    .css-1l02g0t {  /* The class name for the sidebar in Streamlit */
        padding: 0 !important;  /* Remove padding from the sidebar */
    }
    .css-1v0mbdj {  /* The class name for the main content in Streamlit */
        padding: 0 !important;  /* Remove padding from the main content */
        margin-left: 2rem;  /* Adjust margin to control space between sidebar and main content */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.subheader("Select Options")

# Sidebar options
sidebar_options = ["Home", "Data Analytics", "Model Evaluation Metrics", "About"]
selected_option = st.sidebar.selectbox("Go to", sidebar_options)

# Sidebar content
if selected_option == "Home":
    st.sidebar.write("Welcome to the Carbon Emission Data Analytics application. Please select an industry and a product to view detailed analytics.")
elif selected_option == "Data Analytics":
    st.sidebar.write("Select a product to view its carbon emission analytics.")
elif selected_option == "Model Evaluation Metrics":
    st.sidebar.write("Model Evaluation Metrics provide insights into the performance of the carbon emission prediction model.")
elif selected_option == "About":
    st.sidebar.write("This application predicts the carbon emission of different products based on their category and material. It uses a Linear Regression model trained on synthetic data.")

# Main page content based on sidebar selection
if selected_option == "Home":
    st.write("Welcome to the Carbon Emission Data Analytics application. Please select an industry and a product to view detailed analytics.")

elif selected_option == "Data Analytics":
    col1, col2 = st.columns([2, 1])  # Set the ratio of the columns

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
            weight = st.slider("Select Weight", 1, 100, 50)  # Add weight slider
            new_data = pd.DataFrame({
                'Category': [industry],
                'Material': [selected_product['material']],
                'Weight': [weight]
            })
            new_data_encoded = encoder.transform(new_data[['Category', 'Material']])
            new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoder.get_feature_names_out(['Category', 'Material']))
            new_data_encoded_df['Weight'] = new_data['Weight'].values  # Add weight to the encoded data

            # Ensure the columns are in the same order as X_train
            new_data_encoded_df = new_data_encoded_df.reindex(columns=X_train.columns, fill_value=0)

            predicted_emission = model.predict(new_data_encoded_df)
            st.write(f"Predicted Carbon Emission: {predicted_emission[0]:.2f} units")
            
            # Display carbon emission distribution for selected industry and material
            industry_data = df[df['Category'] == industry]
            material_data = industry_data[industry_data['Material'] == selected_product['material']]
            
            # Carbon emission histogram
            st.write("Carbon Emission Distribution for Selected Product:")
            hist_chart = alt.Chart(material_data).mark_bar().encode(
                alt.X("CarbonEmission", bin=True, title="Carbon Emission"),
                alt.Y("count()", title="Count"),
                tooltip=["CarbonEmission", "count()"]
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(hist_chart)
            
            # Scatter plot of Carbon Emission vs. Weight
            st.write("Carbon Emission vs. Weight:")
            scatter_chart = alt.Chart(material_data).mark_circle(size=60).encode(
                x="Weight",
                y="CarbonEmission",
                color="Weight",
                tooltip=["Weight", "CarbonEmission"]
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(scatter_chart)
            
            # Box plot of Carbon Emission
            st.write("Box Plot of Carbon Emission:")
            box_plot = alt.Chart(material_data).mark_boxplot().encode(
                y="CarbonEmission",
                tooltip=["CarbonEmission"]
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(box_plot)

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
