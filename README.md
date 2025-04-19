# 🌾 Agricultural Crop Price Prediction System

## 📋 Overview

An interactive data-driven application that predicts agricultural crop prices across Indian states using machine learning, helping farmers and traders make informed decisions based on historical data and current market trends.

<details>
<summary><b>🎯 Project Objectives</b></summary>

- Provide accurate price predictions for agricultural crops
- Analyze historical price patterns with seasonal variations
- Incorporate market factors into predictions
- Offer actionable insights and recommendations
- Present forecasts in an intuitive format for stakeholders
</details>

## 💻 Technology Stack

<details open>
<summary><b>Core Technologies</b></summary>

- **Python 3.8+**: Core programming language
- **Streamlit**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization
- **Pickle**: Model persistence
</details>

<details>
<summary><b>ML Implementation</b></summary>

- **Random Forest Regressor**: Main prediction algorithm
- **Feature Engineering**: Creating meaningful predictors
- **Model Evaluation**: RMSE and R² metrics
- **Cross-validation**: For model robustness
</details>

## 🚀 Key Features

<details>
<summary><b>Data Exploration</b></summary>

- Upload and analyze agricultural price datasets (CSV/Excel)
- View data summary statistics and distributions
- Visualize price trends over time
- Identify data quality issues and missing values
- Explore relationships between variables
</details>

<details>
<summary><b>Model Training</b></summary>

- Select relevant features for prediction
- Configure model parameters
- Train Random Forest regression models
- Evaluate model performance with key metrics
- Visualize feature importance rankings
- Save models for future predictions
</details>

<details>
<summary><b>Price Prediction</b></summary>

- Select state and crop combinations
- Input environmental factors (rainfall, temperature, soil moisture)
- View current market trends for selected crop
- Get price predictions with confidence measures
- View 7-day price forecasts with trend analysis
- Receive recommendations based on predictions
</details>

<details>
<summary><b>Market Insights</b></summary>

- Browse current market trends for major crops
- Access region-specific market highlights
- View seasonal price patterns
- Get policy impact information
- Stay updated with agricultural news
- Receive tailored recommendations
</details>

## 🗺️ Coverage

<details>
<summary><b>States and Crops Coverage</b></summary>

| Region | States Covered | Major Crops |
|--------|----------------|-------------|
| North | Punjab, Haryana, Uttar Pradesh, Rajasthan, Uttarakhand | Wheat, Rice, Sugarcane, Cotton, Barley |
| South | Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana | Rice, Coffee, Spices, Coconut, Sugarcane |
| East | West Bengal, Bihar, Odisha, Jharkhand, Assam | Rice, Jute, Tea, Maize, Potatoes |
| West | Maharashtra, Gujarat, Madhya Pradesh, Chhattisgarh, Himachal Pradesh | Cotton, Jowar, Soybean, Groundnut, Apples |

</details>

## 📊 Data Format

<details>
<summary><b>Required Data Columns</b></summary>

```
State, District, Market, Crop, Variety, Date, Price, Rainfall, Temperature, Soil_Moisture, Humidity
```

**Sample Data Row:**
```
Maharashtra, Pune, Pune Market, Wheat, Common, 2024-01-15, 2200, 120, 28, 75, 65
```
</details>

## ⚙️ Quick Start

<details>
<summary><b>Installation Steps</b></summary>

```bash
# Clone repository
git clone https://github.com/YourUsername/Crop-Price-Prediction.git
cd Crop-Price-Prediction

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run agri_price_prediction.py
```
</details>

## 🔍 How It Works

<details>
<summary><b>Prediction Methodology</b></summary>

1. **Data Preprocessing**: Clean data, handle missing values, encode categories
2. **Feature Engineering**: Extract time features, create seasonal indicators
3. **Model Training**: Train Random Forest on historical data
4. **Prediction**: Generate base price prediction
5. **Market Adjustment**: Apply current trend factors
6. **Forecast Generation**: Create 7-day projections with trend analysis
</details>

## 🔮 Future Roadmap

<details>
<summary><b>Planned Enhancements</b></summary>

- Weather API integration for automated inputs
- LSTM implementation for time series forecasting
- Mobile application development
- Multi-language support
- SMS price alerts
- Government MSP data integration
- Satellite imagery analysis
</details>

## 🤝 Quick Contribution

<details>
<summary><b>How to Contribute</b></summary>

1. Fork repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit Pull Request

**Development Guidelines:**
- Follow PEP 8 coding standards
- Include docstrings for functions
- Add appropriate comments
- Write tests for new features
</details>

## 📮 Contact & Support

For questions, feedback, or support, please [open an issue](https://github.com/YourUsername/Crop-Price-Prediction/issues) or contact project maintainers.

---

<div align="center">
<p>Made with ❤️ for Indian agricultural stakeholders</p>
</div>
