import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF
from fpdf import FPDF
import uuid
from PIL import Image
import os


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
        {"name": "Cotton Shirt 👕", "material": "Cotton", "image": "images/cotton_shirt.jpg"},
        {"name": "Polyester Jacket 🧥", "material": "Polyester", "image": "images/polyester_jacket.webp"},
        {"name": "Wool Sweater 🧶", "material": "Wool", "image": "images/wool_sweater.jpeg"},
        {"name": "Silk Scarf 🧣", "material": "Silk", "image": "images/silk_scarf.jpeg"},
        {"name": "Denim Jeans 👖", "material": "Polyester", "image": "images/denim_jeans.jpg"},
        {"name": "Leather Jacket 🧥", "material": "Nylon", "image": "images/leather_jacket.jpg"},
        {"name": "Linen Shirt 👔", "material": "Nylon", "image": "images/linen_shirt.webp"},
        {"name": "Nylon Shorts 🩳", "material": "Nylon", "image": "images/nylon_shorts.jpg"}
    ],
    "Automobile": [
        {"name": "Diesel Car 🚗", "material": "Diesel", "image": "images/diesel_car.jpeg"},
        {"name": "Electric Car 🚙", "material": "Electric", "image": "images/electric_car.jpg"},
        {"name": "Petrol Car 🚗", "material": "Petrol", "image": "images/petrol_car.webp"},
        {"name": "Hybrid Car 🚘", "material": "Hybrid", "image": "images/hybrid_car.webp"},
        {"name": "SUV 🚙", "material": "Diesel", "image": "images/suv.jpg"},
        {"name": "Sedan 🚗", "material": "Petrol", "image": "images/sedan.jpg"}
    ],
    "Food": [
        {"name": "Organic Vegetables 🥕", "material": "Vegetables", "image": "images/organic_vegetables.jpg"},
        {"name": "Dairy Milk 🥛", "material": "Dairy", "image": "images/dairy_milk.jpeg"}
    ],
    "Electronics": [
        {"name": "LED Television 📺", "material": "Television", "image": "images/led_television.jpg"},
        {"name": "Smartphone 📱", "material": "Tablet", "image": "images/smartphone.jpg"}
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





# Function to generate PDF report


def remove_emojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')

# Function to generate a PDF report
def generate_pdf_report(product_name, green_score, carbon_emission):

    
    # Save the Altair chart as an image

    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Product Green Score Report", ln=True, align="C")
    
    # Unique ID
    unique_id = str(uuid.uuid4())
    pdf.cell(200, 10, txt=f"Report ID: {unique_id}", ln=True, align="C")
    
    # Product Info
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Product Name: {remove_emojis(product_name)}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Green Score: {green_score}", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Carbon Emission: {carbon_emission:.2f} units", ln=True, align="L")
    
    # Save PDF
    pdf_filename = f"{remove_emojis(product_name).replace(' ', '_')}_report_{unique_id}.pdf"
    pdf.output(pdf_filename)
    
    return pdf_filename

# Streamlit application
st.title('Carbon Emission Data Analytics 📊')

# Add custom CSS to create a fixed box on the right
st.markdown("""
    <style>
    .fixed-box {
        position: fixed;
        top: 30%;
        right: 2%;
        width: 400px;
        Height:400px;
        padding: 30px;
        background-color: #282828;
        box-shadow: 3px 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, ..sidebar .sidebar-content h3, .sidebar .sidebar-content h4 {
        color: #003366;
    }
    .sidebar .sidebar-content p, .sidebar .sidebar-content a {
        color: #666;
    }
    .sidebar .sidebar-content p:hover, .sidebar .sidebar-content a:hover {
        color: #000;
    }
    .sidebar .stSelectbox label {
        font-weight: bold;
        color: #333;
    }
    .sidebar .stSelectbox .st-bo {
        border-color: #003366;
    }
    .sidebar .stButton button {
        background-color: #003366;
        color: white;
        border-radius: 5px;
    }
    .sidebar .stButton button:hover {
        background-color: #005bb5;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("Green_house_Real_Estate_Logo__3.png", width=270)
st.sidebar.title("🌱 Navigation")
st.sidebar.subheader("🔍 Select Options")

# Sidebar options
sidebar_options = ["Home 🏠", "Data Analytics 📊", "Model Evaluation Metrics 🧮", "About ℹ️"]
selected_option = st.sidebar.selectbox("Go to", sidebar_options)

# Sidebar content
if selected_option == "Home 🏠":
    st.sidebar.write("🌍 Welcome to the Carbon Emission Data Analytics application. Please select an industry and a product to view detailed analytics. 🌱")
elif selected_option == "Data Analytics 📊":
    st.sidebar.write("♻️🌍 Select a product to view its carbon emission analytics.")
elif selected_option == "Model Evaluation Metrics 📊":
    st.sidebar.write("Model Evaluation Metrics provide insights into the performance of the carbon emission prediction model.")
elif selected_option == "About ℹ️":
    st.sidebar.write("This application predicts the carbon emission of different products based on their category and material. It uses a Linear Regression model trained on synthetic data.")

# Main page content based on sidebar selection
if selected_option == "Home 🏠":
    st.write("Welcome to the Carbon Emission Data Analytics application. Please select an industry and a product to view detailed analytics.")

elif selected_option == "Data Analytics 📊":
    col1, col2 = st.columns([2, 1])  # Set the ratio of the columns

    with col1:
        # Select industry
        industry = st.selectbox("Select an Industry", list(products.keys()))

        # Select product
        product = st.selectbox("Select a Product", [p["name"] for p in products[industry]])

        # Get selected product's information
        selected_product = next((p for p in products[industry] if p["name"] == product), None)
        selected_product_name = selected_product["name"]
        selected_product_image = selected_product["image"]

        # Render product image
        st.image(selected_product_image, width=200)

        # Filter the input data for the selected product
        input_data = df_encoded[(df_encoded['Category_' + industry] == 1) & (df_encoded['Material_' + selected_product['material']]
        == 1)]

        # Predict carbon emission
        predicted_emission = model.predict(input_data.drop('CarbonEmission', axis=1).iloc[0:1])[0]

        # Calculate green score
        green_score = calculate_green_score(predicted_emission)

        # Get star rating based on green score
        star_rating = get_star_emojis(green_score)

        # Render the fixed box with carbon emission and green score
        render_fixed_box(predicted_emission, green_score, star_rating)

        # Display carbon emission graph
        st.subheader("Carbon Emission Trend 📈")
        chart_data = pd.DataFrame({
            'Material': df['Material'].unique(),
            'Carbon Emission': [model.predict(df_encoded[df_encoded['Material_' + mat] == 1].drop('CarbonEmission', axis=1)).mean() for mat in df['Material'].unique()]
        })

        # Create the graph using Altair
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='Material',
            y='Carbon Emission',
            color='Material'
        ).properties(
            title=f'Average Carbon Emission by Material for {industry} Industry'
        )
        st.altair_chart(chart, use_container_width=True)

        # Add download button for the PDF report
         # Add a button to generate the PDF report
        if st.button("Generate PDF Report"):
            pdf_filename = generate_pdf_report(selected_product_name, green_score, predicted_emission)
            st.success(f"PDF Report Generated: {pdf_filename}")
            with open(pdf_filename, "rb") as file:
                st.download_button(label="Download PDF", data=file, file_name=pdf_filename)

    # with col2:
    #     st.markdown("---")  # Horizontal line separator
    #     st.write("📝 **Product Analysis Summary**")
    #     st.write(f"**Product Name:** {selected_product_name}")
    #     st.write(f"**Predicted Carbon Emission:** {predicted_emission:.2f} units")
    #     st.write(f"**Green Score:** {green_score} {star_rating}")

elif selected_option == "Model Evaluation Metrics 🧮":
    st.write("### Model Evaluation Metrics 🧮")
    st.write("Here are the evaluation metrics for the carbon emission prediction model:")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display the metrics
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")

    # Display residuals graph
    residuals = y_test - y_pred
    residuals_chart = alt.Chart(pd.DataFrame({
        'Actual': y_test,
        'Residuals': residuals
    })).mark_circle(size=60).encode(
        x='Actual',
        y='Residuals',
        color=alt.condition(
            alt.datum.Residuals > 0,  # Highlight positive residuals
            alt.value('green'),
            alt.value('red')
        ),
        tooltip=['Actual', 'Residuals']
    ).properties(
        title="Residuals Plot"
    ).interactive()

    st.altair_chart(residuals_chart, use_container_width=True)

elif selected_option == "About ℹ️":
    st.write("### About ℹ️ This Application")
    st.write("""
        This application predicts the carbon emission of various products based on their industry and material composition.
        It uses a Linear Regression model trained on synthetic data to make these predictions.
        The goal is to help industries and consumers make more informed decisions regarding the environmental impact of their products.
    """)
