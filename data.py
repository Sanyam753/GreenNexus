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
        {"name": "Dairy Milk", "material": "Dairy", "image": "images/dairy_milk.jpeg"},
        {"name": "Grains", "material": "Grains", "image": "images/dairy_milk.jpeg"}
    ],
    "Electronics": [
        {"name": "LED Television", "material": "Television", "image": "images/led_television.jpg"},
        {"name": "Smartphone", "material": "Tablet", "image": "images/smartphone.jpg"}
    ]
}

# Function to calculate the green score
def calculate_green_score(carbon_emission):
    max_emission = df['CarbonEmission'].max()
    min_emission = df['CarbonEmission'].min()
    score = 5 - ((carbon_emission - min_emission) / (max_emission - min_emission) * 4)
    return max(0, round(score, 1))

# Function to get star emojis based on score
def get_star_emojis(score):
    full_stars = int(score)
    half_star = int((score - full_stars) >= 0.5)
    empty_stars = 5 - full_stars - half_star
    return '⭐' * full_stars + '✰' * half_star + '☆' * empty_stars

# Function to render the fixed box content
def render_fixed_box(predicted_emission=None, green_score=None, star_rating=None):
    st.markdown("""
        <div class="fixed-box">
            <h4>Product Analysis</h4>
            <p><b>Predicted Carbon Emission:</b> {0:.2f} units</p>
            <p><b>Green Score:</b> {1} {2}</p>
        </div>
    """.format(predicted_emission if predicted_emission is not None else 0,
               green_score if green_score is not None else 0,
               star_rating if star_rating is not None else ''), unsafe_allow_html=True)

# Streamlit application
st.title('Carbon Emission Data Analytics')

# Add custom CSS to create a fixed box on the right
st.markdown("""
    <style>
    .fixed-box {
        position: fixed;
        top: 30%;
        right: 2%;
        width: 350px;
        Height:350px;
        padding: 30px;
        background-color: #282828;
        box-shadow: 3px 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
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

            predicted_emission = model.predict(new_data_encoded_df)[0]
            green_score = calculate_green_score(predicted_emission)
            star_rating = get_star_emojis(green_score)

            # Render the fixed box with the prediction and green score
            render_fixed_box(predicted_emission, green_score, star_rating)

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

            # Display specific product information
            st.image(selected_product["image"], width=150)  # Set fixed width
            st.write(f"Material: {selected_product['material']}")
            st.write(f"Industry: {industry}")

elif selected_option == "Model Evaluation Metrics":
    st.header("Model Evaluation Metrics")
    y_pred = model.predict(X_test)

    st.subheader("Mean Squared Error (MSE)")
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"MSE: {mse:.2f}")

    st.subheader("Mean Absolute Error (MAE)")
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"MAE: {mae:.2f}")

    st.subheader("R-squared (R2)")
    r2 = r2_score(y_test, y_pred)
    st.write(f"R2: {r2:.2f}")

    # Histogram of residuals
    st.subheader("Residuals Histogram")
    residuals = y_test - y_pred
    residuals_chart = alt.Chart(pd.DataFrame(residuals, columns=['Residuals'])).mark_bar().encode(
        alt.X("Residuals", bin=True, title="Residuals"),
        alt.Y("count()", title="Count"),
        tooltip=["Residuals", "count()"]
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(residuals_chart)

    # Scatter plot of actual vs. predicted values
    st.subheader("Actual vs. Predicted Values")
    scatter_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    scatter_chart = alt.Chart(scatter_data).mark_circle(size=60).encode(
        x="Actual",
        y="Predicted",
        color="Predicted",
        tooltip=["Actual", "Predicted"]
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(scatter_chart)

elif selected_option == "About":
    st.header("About")
    st.write("""
    This application predicts the carbon emission of different products based on their category and material.
    It uses a Linear Regression model trained on synthetic data to make predictions.
    The application also provides various data analytics and visualization tools to help users understand the carbon emission patterns of different products.
    """)
    st.write("Contact: Sanyam Sankhala: sanyamsankhala753@gmail.com")