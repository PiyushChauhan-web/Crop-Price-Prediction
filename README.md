🌾 Agricultural Crop Price Prediction System
📋 Overview
An interactive data-driven application that predicts agricultural crop prices across Indian states using machine learning. It helps farmers and traders make informed decisions based on historical data and current market trends.

🎯 Project Objectives
Provide accurate price predictions for agricultural crops

Analyze historical price patterns with seasonal variations

Incorporate market factors into predictions

Offer actionable insights and recommendations

Present forecasts in an intuitive format for stakeholders

💻 Technology Stack
Core Technologies:

Python 3.8+

Streamlit

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

Pickle

Machine Learning Implementation:

Random Forest Regressor

Feature Engineering

Model Evaluation using RMSE and R²

Cross-validation

🚀 Key Features
1. Data Exploration

Upload and analyze agricultural price datasets (CSV/Excel)

View data summary statistics

Visualize price trends over time

Identify missing or inconsistent data

Explore relationships between variables

2. Model Training

Select features and configure model

Train Random Forest regression models

Evaluate with RMSE and R²

Visualize feature importance

Save trained models

3. Price Prediction

Choose state and crop

Enter rainfall, temperature, soil moisture, etc.

View current market trends

Get price predictions with confidence

Access 7-day forecasts and recommendations

4. Market Insights

View current crop trends

Access region-specific highlights

Analyze seasonal price patterns

Get updates on policy impact and news

🗺️ States and Crops Coverage

Region	States Covered	Major Crops
North	Punjab, Haryana, Uttar Pradesh, Rajasthan, Uttarakhand	Wheat, Rice, Sugarcane, Cotton, Barley
South	Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana	Rice, Coffee, Spices, Coconut, Sugarcane
East	West Bengal, Bihar, Odisha, Jharkhand, Assam	Rice, Jute, Tea, Maize, Potatoes
West	Maharashtra, Gujarat, Madhya Pradesh, Chhattisgarh, Himachal Pradesh	Cotton, Jowar, Soybean, Groundnut, Apples
📊 Required Data Format
Columns:

javascript
Copy
Edit
State, District, Market, Crop, Variety, Date, Price, Rainfall, Temperature, Soil_Moisture, Humidity
Sample Row:

yaml
Copy
Edit
Maharashtra, Pune, Pune Market, Wheat, Common, 2024-01-15, 2200, 120, 28, 75, 65
⚙️ Quick Start
bash
Copy
Edit
# Clone the repository
git clone https://github.com/YourUsername/Crop-Price-Prediction.git
cd Crop-Price-Prediction

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run agri_price_prediction.py
🔍 How It Works
Data Preprocessing – Clean data, handle nulls, encode categories

Feature Engineering – Extract time-based and seasonal features

Model Training – Train Random Forest on historical data

Prediction – Generate predictions and apply trend adjustments

Forecasting – Show 7-day projections with visual analysis

🔮 Future Roadmap
Weather API for automated inputs

LSTM model for better time-series forecasting

Mobile application version

Multi-language support

SMS-based alerts

Government MSP data integration

Satellite image analysis for better predictions

🤝 How to Contribute
Fork the repository

Create a branch: git checkout -b feature/new-feature

Commit changes: git commit -m "Add new feature"

Push the branch: git push origin feature/new-feature

Open a Pull Request

Guidelines:

Follow PEP 8

Include docstrings

Add relevant comments

Write test cases

📮 Contact & Support
For questions or support, please open an issue here.

<p align="center">Made with ❤️ for Indian agricultural stakeholders</p>
