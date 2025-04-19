# 🌾 Agricultural Crop Price Prediction System

![Project Banner](https://www.researchgate.net/publication/349649635/figure/fig2/AS:995564511141888@1614324310627/Crop-price-prediction-steps.png)

## 📋 Overview

The Agricultural Crop Price Prediction System is a data-driven application designed to help farmers, traders, and policymakers predict crop prices based on historical data and current market trends. This tool provides valuable insights that can aid in making informed decisions about when to sell crops, what crops to grow, and how to plan for market fluctuations.

<details>
<summary><b>🎯 Click to view project objectives</b></summary>

- Provide accurate price predictions for different agricultural crops across Indian states
- Analyze historical price trends and patterns
- Incorporate seasonal variations and market factors into predictions
- Offer market insights and recommendations based on current trends
- Present forecasts in an intuitive, easy-to-understand format for farmers and stakeholders
</details>

## 🚀 Live Demo

[Explore the application](https://crop-price-prediction.streamlit.app/) *(Demo link - replace with actual when deployed)*

## 💻 Technologies Used

<details open>
<summary><b>Tech Stack</b></summary>

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework for creating interactive data apps
- **Pandas & NumPy**: Data manipulation and numerical computations
- **Scikit-learn**: Machine learning algorithms for price prediction
- **Matplotlib & Seaborn**: Data visualization
- **Pickle**: Model serialization and deserialization
</details>

<details>
<summary><b>Machine Learning</b></summary>

- **Random Forest Regressor**: Primary prediction model
- **Feature Engineering**: Transforming raw data into meaningful features
- **Model Evaluation**: Using metrics like RMSE and R² score
- **Cross-validation**: Ensuring model robustness
</details>

## 📊 Features

### 1. Data Exploration
![Data Exploration](https://i.imgur.com/JGf3m9Q.png)
- Upload and analyze agricultural price datasets
- View data summary statistics
- Visualize price distributions and trends
- Check for missing values and data quality

### 2. Model Training
![Model Training](https://i.imgur.com/BvdDE1F.png)
- Select features for training
- Configure model parameters
- Train Random Forest regression models
- Evaluate model performance with metrics
- Visualize feature importance

### 3. Price Prediction
![Price Prediction](https://i.imgur.com/RSdEoGq.png)
- Select state and crop
- Input features like rainfall, temperature, and soil moisture
- View current market trends for selected crop
- Get price predictions with confidence intervals
- See 7-day price forecasts with trend visualization

### 4. Market Insights
![Market Insights](https://i.imgur.com/NJW5fQh.png)
- Browse current market trends for major crops
- Get region-specific market highlights
- View recommendations based on predicted trends
- Stay updated with agricultural news and policies

## 📸 Application Screenshots

<details>
<summary><b>Click to view more screenshots</b></summary>

### Home Page
![Home Page](https://i.imgur.com/pZXc2BL.png)
*The welcoming interface of the application with navigation options*

### Feature Importance
![Feature Importance](https://i.imgur.com/YThvxu1.png)
*Visualization of which factors most influence crop prices*

### Price Forecast Chart
![Price Forecast](https://i.imgur.com/ZxSWNvK.png)
*7-day price forecast visualization with trend lines*

### Regional Market Highlights
![Regional Insights](https://i.imgur.com/FWGDnjr.png)
*State and crop specific market insights*
</details>

## 🗺️ Coverage

The system covers 20 states across India with their specific crop mappings:

<details>
<summary><b>Click to see covered states and crops</b></summary>

| State | Major Crops |
|-------|------------|
| Andhra Pradesh | Rice, Cotton, Chillies, Turmeric, Sugarcane |
| Assam | Rice, Tea, Jute, Sugarcane, Oilseeds |
| Bihar | Rice, Wheat, Maize, Pulses, Sugarcane |
| Chhattisgarh | Rice, Maize, Pulses, Oilseeds, Wheat |
| Gujarat | Cotton, Groundnut, Wheat, Bajra, Sugarcane |
| Haryana | Wheat, Rice, Sugarcane, Cotton, Oilseeds |
| Karnataka | Rice, Ragi, Jowar, Coffee, Sugarcane |
| Kerala | Coconut, Rice, Rubber, Spices, Banana |
| Madhya Pradesh | Wheat, Soybean, Pulses, Rice, Cotton |
| Maharashtra | Jowar, Cotton, Sugarcane, Soybean, Rice |
| Odisha | Rice, Pulses, Oilseeds, Jute, Sugarcane |
| Punjab | Wheat, Rice, Cotton, Sugarcane, Maize |
| Rajasthan | Wheat, Barley, Pulses, Oilseeds, Cotton |
| Tamil Nadu | Rice, Sugarcane, Coconut, Cotton, Groundnut |
| Telangana | Rice, Cotton, Maize, Pulses, Chillies |
| Uttar Pradesh | Wheat, Sugarcane, Rice, Pulses, Potato |
| West Bengal | Rice, Jute, Potato, Tea, Oilseeds |
| Himachal Pradesh | Apple, Wheat, Maize, Potato, Ginger |
| Jharkhand | Rice, Maize, Pulses, Wheat, Oilseeds |
| Uttarakhand | Rice, Wheat, Pulses, Oilseeds, Sugarcane |
</details>

## 📊 Data Requirements

The application works with agricultural price datasets containing the following recommended columns:

- State
- District
- Market
- Crop
- Variety
- Date
- Price
- Rainfall
- Temperature
- Soil_Moisture
- Humidity

<details>
<summary><b>Click to see sample data format</b></summary>

| State | District | Market | Crop | Variety | Date | Price | Rainfall | Temperature | Soil_Moisture |
|-------|----------|--------|------|---------|------|-------|----------|-------------|--------------|
| Maharashtra | Pune | Pune Mkt | Wheat | Common | 2024-01-15 | 2200 | 120 | 28 | 75 |
| Punjab | Ludhiana | Ludhiana Mkt | Rice | Basmati | 2024-01-16 | 3500 | 95 | 32 | 65 |
| Karnataka | Mysore | Mysore Mkt | Ragi | Local | 2024-01-17 | 1800 | 85 | 30 | 60 |
| Tamil Nadu | Coimbatore | Coimbatore Mkt | Sugarcane | Co-86032 | 2024-01-18 | 310 | 110 | 33 | 70 |
| Gujarat | Ahmedabad | Ahmedabad Mkt | Cotton | Long Staple | 2024-01-19 | 6500 | 45 | 36 | 50 |
</details>

## 🚀 Installation & Setup

<details>
<summary><b>Click to see installation instructions</b></summary>

```bash
# Clone the repository
git clone https://github.com/YourUsername/Crop-Price-Prediction.git
cd Crop-Price-Prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run agri_price_prediction.py
```

### Requirements
Create a `requirements.txt` file with the following dependencies:

```
streamlit==1.31.0
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.0
seaborn==0.13.0
```
</details>

## 🔍 How It Works

<details>
<summary><b>Click to see how the prediction works</b></summary>

1. **Data Collection**: The system uses historical agricultural price data combined with weather parameters.

2. **Preprocessing**: Data is cleaned, missing values are handled, and categorical features are encoded.

3. **Feature Engineering**: The system extracts useful features from date columns and creates seasonal indicators.

4. **Model Training**: A Random Forest Regressor is trained on the processed data.

5. **Prediction**: For new inputs, the model predicts the crop price based on the provided features.

6. **Market Adjustment**: Predictions are adjusted based on current market trends and seasonal patterns.

7. **Forecast Generation**: The system generates a 7-day forecast using the base prediction and trend factors.
</details>

## 💡 Future Enhancements

<details>
<summary><b>Click to see planned features</b></summary>

- Integration with real-time weather API for automated inputs
- Implementation of more advanced models like LSTM for time series forecasting
- Mobile application for farmers with limited internet access
- Multi-language support for regional languages
- SMS alerts for significant price changes
- Integration with government MSP and procurement data
- Satellite imagery analysis for crop health assessment
</details>

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

<details>
<summary><b>Click to see contribution guidelines</b></summary>

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
</details>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue or contact the project maintainers.

---

<div align="center">
<p>Made with ❤️ for Indian farmers and agricultural stakeholders</p>
</div>
