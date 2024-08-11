import streamlit as st
import joblib
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from io import BytesIO
import uuid

# Function to generate PDF report
def generate_pdf_report(report_id, green_score, carbon_emissions, data_summary, tips):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Custom styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    title_style.fontName = 'Helvetica-Bold'
    title_style.fontSize = 20
    title_style.textColor = '#003366'

    heading_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor='#003366',
        spaceAfter=15
    )

    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        textColor='#000000',
        spaceAfter=11
    )

    # Add margins
    margin = 0.75 * inch

    # Title
    c.setFont('Helvetica-Bold', 20)
    c.setFillColor('#003366')
    c.drawString(margin, height - margin, "Enterprise Carbon Footprint & Green Score Report")

    # Report ID
    c.setFont('Helvetica', 12)
    c.setFillColor('#000000')
    c.drawString(margin, height - margin - 30, f"Report ID: {report_id}")

    # Green Score
    c.setFont('Helvetica-Bold', 18)
    c.setFillColor('#003366')
    c.drawString(margin, height - 2.5 * inch, "Green Score 🌿")
    c.setFont('Helvetica', 14)
    c.drawString(margin, height - 2.75 * inch, f"Predicted Green Score: {green_score:.2f} / 10")

    # Carbon Emissions
    c.setFont('Helvetica-Bold', 18)
    c.drawString(margin, height - 3.5 * inch, "Carbon Emissions 🌍")
    c.setFont('Helvetica', 14)
    c.drawString(margin, height - 3.75 * inch, f"Predicted Carbon Emissions: {carbon_emissions:.2f} kg CO2e/month")

    # Data Summary
    c.setFont('Helvetica-Bold', 18)
    c.drawString(margin, height - 4.5 * inch, "Data Summary 📊")
    text = "\n".join([f"{key}: {value}" for key, value in data_summary.items()])
    c.setFont('Helvetica', 12)
    text_object = c.beginText(margin, height - 4.75 * inch)
    text_object.setTextOrigin(margin, height - 4.75 * inch)
    text_object.setFont("Helvetica", 12)
    text_object.textLines(text)
    c.drawText(text_object)

    # Tips for Improvement
    c.setFont('Helvetica-Bold', 18)
    c.drawString(margin, height - 8 * inch, "Tips for Improvement 💡")
    text = "\n".join(tips)
    c.setFont('Helvetica', 12)
    text_object = c.beginText(margin, height - 8.25 * inch)
    text_object.setTextOrigin(margin, height - 8.25 * inch)
    text_object.setFont("Helvetica", 12)
    text_object.textLines(text)
    c.drawText(text_object)

    # Save the PDF
    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.getvalue()

# Load the trained models
model_green = joblib.load('model_green_extended.pkl')
model_carbon = joblib.load('model_carbon_extended.pkl')

# Function to display custom CSS
def apply_custom_css():
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background-color: #003366;
            color: white;
        }
        .sidebar .sidebar-content h1 {
            color: #ffffff;
        }
        .title {
            font-size: 36px;
            color: #003366;
            font-weight: bold;
        }
        .subheader {
            font-size: 24px;
            color: #003366;
            font-weight: bold;
            padding:20px;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #00509e;
        }
        .stMarkdown>div {
            font-size: 18px;
        }
        .result-card {
            border: 1px solid #003366;
            padding: 30px;
            border-radius: 10px;
            background-color: #40534C;
            box-shadow: 3px 4px 6px rgb(27, 66, 66);
            margin-bottom: 20px;
        }
        .tips-card {
            border: 1px solid #003366;
            padding: 30px;
            border-radius: 10px;
            background-color: #1A3636;
            box-shadow: 5px 4px 8px rgb(27, 66, 66);
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply custom CSS
apply_custom_css()

# Title and description
st.title("Enterprise Carbon Footprint & Green Score Calculator 🌱")
st.write("Welcome to the Enterprise Carbon Footprint and Green Score Calculator. Enter your data below to calculate your company's Green Score and Carbon Emissions.")

# Sidebar with logo and inputs
logo = "images/Green_score_logo2.png"  # Update this path to your logo file
st.sidebar.image(logo, use_column_width=True, width=50)  # Adjust the width as needed
st.sidebar.title("Enter Your Data")
st.sidebar.write("Use the options below to enter your enterprise's data.")

# Create input fields for all the parameters with increased step values
energy_consumption = st.sidebar.number_input('Energy Consumption (kWh/month)', min_value=0.0, step=100.0, format="%.2f")
waste_generated = st.sidebar.number_input('Waste Generated (kg/month)', min_value=0.0, step=100.0, format="%.2f")
recycling_rate = st.sidebar.slider('Recycling Rate (%)', min_value=0, max_value=100, value=50, step=5) / 100.0
green_investments = st.sidebar.number_input('Investments in Green Projects (USD)', min_value=0.0, step=1000.0, format="%.2f")
water_consumption = st.sidebar.number_input('Water Consumption (Liters/month)', min_value=0.0, step=100.0, format="%.2f")
transportation_emissions = st.sidebar.number_input('Transportation Emissions (kg CO2e/month)', min_value=0.0, step=100.0, format="%.2f")
renewable_energy_usage = st.sidebar.slider('Renewable Energy Usage (%)', min_value=0, max_value=100, value=50, step=1) / 100.0
employee_awareness = st.sidebar.number_input('Employee Awareness Programs (Hours/month)', min_value=0.0, step=10.0, format="%.2f")

# Create a button to trigger the calculation
if st.sidebar.button('Calculate'):
    # Prepare the input data for prediction
    input_data = np.array([[energy_consumption, waste_generated, recycling_rate, green_investments, 
                            water_consumption, transportation_emissions, renewable_energy_usage, 
                            employee_awareness]])

    # Predict the Green Score and Carbon Emissions
    carbon_emissions = model_carbon.predict(input_data)[0]
    
    # Inverse relation for green score: a simple linear inverse relation (example)
    green_score = 1000 / (1 + carbon_emissions)  # Adjust this formula as needed

    # Generate unique report ID
    report_id = str(uuid.uuid4())

    # Prepare data summary and tips
    data_summary = {
        "Energy Consumption": f"{energy_consumption} kWh/month",
        "Waste Generated": f"{waste_generated} kg/month",
        "Recycling Rate": f"{recycling_rate * 100}%",
        "Investments in Green Projects": f"${green_investments:.2f}",
        "Water Consumption": f"{water_consumption} Liters/month",
        "Transportation Emissions": f"{transportation_emissions} kg CO2e/month",
        "Renewable Energy Usage": f"{renewable_energy_usage * 100}%",
        "Employee Awareness Programs": f"{employee_awareness} Hours/month"
    }


    tips = [
        "🌿 Consider increasing your investments in green projects to further reduce your carbon footprint.",
        "♻️ Improve your recycling rate to minimize waste and enhance resource efficiency.",
        "💧 Explore ways to reduce water consumption to conserve this precious resource.",
        "🚗 Optimize transportation logistics to lower emissions and reduce your environmental impact.",
        "🌍 Increase the use of renewable energy sources to decrease dependence on fossil fuels.",
        "📚 Invest in employee awareness programs to promote sustainability practices within your organization."
    ]

    # Generate and download PDF report
    pdf_report = generate_pdf_report(report_id, green_score, carbon_emissions, data_summary, tips)
    st.download_button(
        label="Download PDF Report 📄",
        data=pdf_report,
        file_name=f"carbon_footprint_report_{report_id}.pdf",
        mime="application/pdf"
    )

    # Display results in cards
    st.markdown(f"""
    <div class="result-card">
        <h2>Green Score 🌿</h2>
        <h4>Predicted Green Score: {green_score:.2f} / 10</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <h2>Carbon Emissions 🌍</h2>
        <h4>Predicted Carbon Emissions: {carbon_emissions:.2f} kg CO2e/month</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <h2>Data Summary 📊</h2>
        <ul>
            {" ".join([f"<li><strong>{key}:</strong> {value}</li>" for key, value in data_summary.items()])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="tips-card">
        <h2>Tips for Improvement 💡</h2>
        <ul>
            {" ".join([f"<li>{tip}</li>" for tip in tips])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
