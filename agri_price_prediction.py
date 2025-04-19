import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import requests
import json
from datetime import datetime
import joblib
import os

class CropPredictionApp:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.data = None
        self.api_key = "YOUR_API_KEY"  # Replace with actual API key for market data
        self.market_api_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"  # Example API

    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def preprocess_data(self):
        """Preprocess the data for model training"""
        # Identify categorical and numerical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target column from features
        if 'crop_price' in numerical_cols:
            numerical_cols.remove('crop_price')
        
        # Create column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the prediction model"""
        if self.data is None:
            return False, "No data loaded. Please load data first."
        
        try:
            # Check if target column exists
            if 'crop_price' not in self.data.columns:
                return False, "Target column 'crop_price' not found in data."
            
            # Split features and target
            X = self.data.drop('crop_price', axis=1)
            y = self.data['crop_price']
            
            # Preprocess data
            preprocessor = self.preprocess_data()
            
            # Create and train model pipeline
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=random_state))
            ])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.model = model
            
            return True, {
                "model": model,
                "metrics": {
                    "MSE": mse,
                    "R-squared": r2
                }
            }
        
        except Exception as e:
            return False, f"Error training model: {str(e)}"
    
    def save_model(self, model_path='crop_prediction_model.pkl'):
        """Save the trained model to disk"""
        if self.model is None:
            return False, "No trained model to save."
        
        try:
            joblib.dump(self.model, model_path)
            return True, f"Model saved to {model_path}"
        except Exception as e:
            return False, f"Error saving model: {str(e)}"
    
    def load_model(self, model_path='crop_prediction_model.pkl'):
        """Load a trained model from disk"""
        try:
            self.model = joblib.load(model_path)
            return True, "Model loaded successfully!"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def predict_price(self, input_data):
        """Predict crop price based on input data"""
        if self.model is None:
            return False, "No trained model available. Please train or load a model first."
        
        try:
            # Convert input data to DataFrame if it's not already
            if not isinstance(input_data, pd.DataFrame):
                input_data = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.model.predict(input_data)
            return True, prediction[0]
        
        except Exception as e:
            return False, f"Error making prediction: {str(e)}"
    
    def get_market_price(self, crop_name, location):
        """Get real-time market price for a crop from API"""
        try:
            # In a real application, you would make an API call here
            # For demonstration purposes, we'll simulate this
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'offset': 0,
                'limit': 10,
                'filters[commodity]': crop_name,
                'filters[state]': location
            }
            
            # Simulated API response for demonstration
            # In a real application, uncomment the following lines:
            # response = requests.get(self.market_api_url, params=params)
            # if response.status_code == 200:
            #     data = response.json()
            #     # Process the data as needed
            #     return True, data
            # else:
            #     return False, f"API error: {response.status_code}"
            
            # Simulated response
            simulated_price = np.random.uniform(20, 100)
            return True, {
                "crop": crop_name,
                "location": location,
                "current_price": simulated_price,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        except Exception as e:
            return False, f"Error fetching market price: {str(e)}"
    
    def analyze_factors(self, input_data):
        """Analyze which factors are most influencing the predicted price"""
        if self.model is None:
            return False, "No trained model available. Please train or load a model first."
        
        try:
            # For RandomForestRegressor, we can use feature importances
            regressor = self.model.named_steps['regressor']
            feature_importances = regressor.feature_importances_
            
            # Get feature names after preprocessing
            preprocessor = self.model.named_steps['preprocessor']
            
            # Try to get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                # If older scikit-learn version
                categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
                numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if 'crop_price' in numerical_cols:
                    numerical_cols.remove('crop_price')
                feature_names = numerical_cols + [f"{col}_{val}" for col in categorical_cols for val in self.data[col].unique()]
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            return True, importance_df
        
        except Exception as e:
            return False, f"Error analyzing factors: {str(e)}"
    
    def visualize_predictions(self, predicted_price, market_price, crop_name):
        """Create visualization comparing predicted vs market price"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        prices = [predicted_price, market_price]
        labels = ['Predicted Price', 'Market Price']
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(labels, prices, color=colors)
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'₹{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
        
        # Add styling
        ax.set_title(f'Price Comparison for {crop_name}', fontsize=15, fontweight='bold')
        ax.set_ylabel('Price (₹)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add differential indicator
        diff = predicted_price - market_price
        diff_percent = (diff / market_price) * 100
        diff_text = f"Difference: ₹{abs(diff):.2f} ({abs(diff_percent):.1f}%)"
        if diff > 0:
            diff_text += " (Predicted higher)"
            diff_color = 'green'
        elif diff < 0:
            diff_text += " (Market higher)"
            diff_color = 'red'
        else:
            diff_text += " (Equal)"
            diff_color = 'gray'
        
        plt.figtext(0.5, 0.01, diff_text, ha="center", fontsize=12, color=diff_color)
        
        plt.tight_layout(pad=3.0)
        return fig


def create_streamlit_app():
    """Create the Streamlit web application"""
    st.set_page_config(page_title="Advanced Crop Price Prediction System", layout="wide")
    
    # Initialize app
    app = CropPredictionApp()
    
    # Add CSS for better UI
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #2c3e50;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
        color: #34495e;
    }
    .info-box {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
    .success-box {
        background-color: #f0fff4;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
    .warning-box {
        background-color: #fff9f0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header
    st.markdown('<p class="big-font">Advanced Crop Price Prediction System</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    This system predicts crop prices based on various factors including weather quality, soil conditions,
    crop type, and other parameters. It also compares the predicted prices with real-time market rates.
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Management", "Model Training", "Prediction", "Market Analysis", "About"])
    
    # Data Management Page
    if page == "Data Management":
        st.markdown('<p class="medium-font">Data Management</p>', unsafe_allow_html=True)
        
        # Upload data
        uploaded_file = st.file_uploader("Upload CSV data file", type=["csv"])
        if uploaded_file is not None:
            success, message = app.load_data(uploaded_file)
            if success:
                st.success(message)
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(app.data.head())
                
                # Show basic statistics
                st.subheader("Data Statistics")
                st.dataframe(app.data.describe())
                
                # Data visualization
                st.subheader("Data Visualization")
                
                # Select columns to visualize
                numeric_cols = app.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if 'crop_price' in numeric_cols:
                    # Correlation with crop price
                    st.write("Correlation with Crop Price")
                    corr = app.data[numeric_cols].corr()['crop_price'].sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    corr.drop('crop_price').plot(kind='bar', ax=ax)
                    plt.title('Correlation with Crop Price')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Distribution of crop prices
                    st.write("Distribution of Crop Prices")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(app.data['crop_price'], kde=True, ax=ax)
                    plt.title('Distribution of Crop Prices')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Missing values information
                st.subheader("Missing Values Information")
                missing = app.data.isnull().sum()
                if missing.sum() > 0:
                    st.dataframe(missing[missing > 0])
                else:
                    st.write("No missing values found in the dataset.")
                
                # Data preparation options
                st.subheader("Data Preparation")
                if st.checkbox("Handle missing values automatically"):
                    # Handle numeric missing values with median
                    for col in app.data.select_dtypes(include=['int64', 'float64']).columns:
                        app.data[col].fillna(app.data[col].median(), inplace=True)
                    
                    # Handle categorical missing values with mode
                    for col in app.data.select_dtypes(include=['object']).columns:
                        app.data[col].fillna(app.data[col].mode()[0], inplace=True)
                    
                    st.success("Missing values handled successfully!")
            else:
                st.error(message)

    # Model Training Page
    elif page == "Model Training":
        st.markdown('<p class="medium-font">Model Training</p>', unsafe_allow_html=True)
        
        if app.data is None:
            st.warning("Please load data first in the Data Management section.")
        else:
            # Model parameters
            st.subheader("Model Parameters")
            test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    success, result = app.train_model(test_size, random_state)
                    
                    if success:
                        st.success("Model trained successfully!")
                        
                        # Show model metrics
                        st.subheader("Model Performance")
                        metrics = result["metrics"]
                        col1, col2 = st.columns(2)
                        col1.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
                        col2.metric("R-squared Score", f"{metrics['R-squared']:.4f}")
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        success, importance = app.analyze_factors(None)
                        if success:
                            # Plot top 10 features
                            fig, ax = plt.subplots(figsize=(10, 6))
                            top_features = importance.head(10)
                            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                            plt.title('Top 10 Most Important Features')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Save model option
                        if st.button("Save Trained Model"):
                            success, message = app.save_model()
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    else:
                        st.error(result)
            
            # Load model option
            st.subheader("Or Load Existing Model")
            model_file = st.file_uploader("Upload model file", type=["pkl"])
            if model_file is not None:
                # Save uploaded model to temp file
                with open("temp_model.pkl", "wb") as f:
                    f.write(model_file.getbuffer())
                
                # Load the model
                success, message = app.load_model("temp_model.pkl")
                if success:
                    st.success(message)
                else:
                    st.error(message)

    # Prediction Page
    elif page == "Prediction":
        st.markdown('<p class="medium-font">Crop Price Prediction</p>', unsafe_allow_html=True)
        
        if app.model is None:
            st.warning("Please train or load a model first.")
        else:
            st.markdown("""
            <div class="info-box">
            Enter the crop details and environmental factors to predict the price.
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different input methods
            tab1, tab2 = st.tabs(["Form Input", "CSV Input"])
            
            with tab1:
                # Get column names from the trained model
                try:
                    # Try to get feature names from the model
                    features = app.data.drop('crop_price', axis=1).columns.tolist()
                    
                    # Create dynamic form based on features
                    st.subheader("Enter Crop Details")
                    
                    # Organize features into categories for better UI
                    soil_features = [f for f in features if 'soil' in f.lower()]
                    weather_features = [f for f in features if any(w in f.lower() for w in ['temp', 'rain', 'humid', 'weather'])]
                    crop_features = [f for f in features if 'crop' in f.lower()]
                    other_features = [f for f in features if f not in soil_features + weather_features + crop_features]
                    
                    # Create form sections
                    col1, col2 = st.columns(2)
                    
                    # Input dictionary
                    input_data = {}
                    
                    # Soil features
                    with col1:
                        st.write("Soil Parameters")
                        for feature in soil_features:
                            if app.data[feature].dtype == 'object':
                                input_data[feature] = st.selectbox(feature, options=app.data[feature].unique())
                            else:
                                input_data[feature] = st.number_input(feature, 
                                                                   value=float(app.data[feature].mean()),
                                                                   step=0.1)
                    
                    # Weather features
                    with col2:
                        st.write("Weather Parameters")
                        for feature in weather_features:
                            if app.data[feature].dtype == 'object':
                                input_data[feature] = st.selectbox(feature, options=app.data[feature].unique())
                            else:
                                input_data[feature] = st.number_input(feature, 
                                                                   value=float(app.data[feature].mean()),
                                                                   step=0.1)
                    
                    # Crop features
                    st.write("Crop Information")
                    cols = st.columns(min(3, len(crop_features)))
                    for i, feature in enumerate(crop_features):
                        with cols[i % min(3, len(crop_features))]:
                            if app.data[feature].dtype == 'object':
                                input_data[feature] = st.selectbox(feature, options=app.data[feature].unique())
                            else:
                                input_data[feature] = st.number_input(feature, 
                                                                   value=float(app.data[feature].mean()),
                                                                   step=0.1)
                    
                    # Other features
                    if other_features:
                        st.write("Other Parameters")
                        cols = st.columns(min(3, len(other_features)))
                        for i, feature in enumerate(other_features):
                            with cols[i % min(3, len(other_features))]:
                                if app.data[feature].dtype == 'object':
                                    input_data[feature] = st.selectbox(feature, options=app.data[feature].unique())
                                else:
                                    input_data[feature] = st.number_input(feature, 
                                                                       value=float(app.data[feature].mean()),
                                                                       step=0.1)
                    
                    # Additional parameters for market comparison
                    st.subheader("Market Comparison")
                    crop_name = st.text_input("Crop Name for Market Comparison", "Rice")
                    location = st.text_input("Location/State", "Punjab")
                    
                    # Predict button
                    if st.button("Predict Price"):
                        # Predict with model
                        success, predicted_price = app.predict_price(input_data)
                        
                        if success:
                            # Get market price
                            market_success, market_data = app.get_market_price(crop_name, location)
                            
                            if market_success:
                                market_price = market_data["current_price"]
                                
                                # Display results
                                st.subheader("Prediction Results")
                                
                                # Create columns for results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="success-box">
                                    <h3>Predicted Price:</h3>
                                    <h2>₹{predicted_price:.2f} per kg</h2>
                                    <p>Based on the provided parameters</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="warning-box">
                                    <h3>Current Market Price:</h3>
                                    <h2>₹{market_price:.2f} per kg</h2>
                                    <p>Source: Market data as of {market_data["timestamp"]}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Price comparison visualization
                                st.subheader("Price Comparison")
                                fig = app.visualize_predictions(predicted_price, market_price, crop_name)
                                st.pyplot(fig)
                                
                                # Insights
                                diff = predicted_price - market_price
                                diff_percent = (diff / market_price) * 100
                                
                                st.subheader("Analysis Insights")
                                if diff > 0 and diff_percent > 10:
                                    st.markdown("""
                                    <div class="success-box">
                                    <h3>Favorable Selling Conditions</h3>
                                    <p>The predicted price is significantly higher than the current market price. 
                                    Consider holding your crop if possible until market conditions improve to match prediction.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif diff < 0 and abs(diff_percent) > 10:
                                    st.markdown("""
                                    <div class="warning-box">
                                    <h3>Challenging Market Conditions</h3>
                                    <p>The predicted price is significantly lower than the current market price.
                                    Consider selling soon as prices may decline based on prediction factors.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="info-box">
                                    <h3>Stable Market Conditions</h3>
                                    <p>The predicted price is close to the current market price.
                                    Market conditions appear stable for this crop.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Factor analysis
                                st.subheader("Factor Influence Analysis")
                                success, importance = app.analyze_factors(input_data)
                                if success:
                                    # Show top 5 factors
                                    st.write("Top factors influencing this prediction:")
                                    st.dataframe(importance.head(5))
                            else:
                                st.error(f"Error getting market price: {market_data}")
                        else:
                                st.error(f"Error making prediction: {predicted_price}")
                
                except Exception as e:
                    st.error(f"Error creating prediction form: {str(e)}")
                    st.write("Please make sure you've trained a model with appropriate data first.")
            
            with tab2:
                st.subheader("Upload CSV for Batch Prediction")
                batch_file = st.file_uploader("Upload CSV with crop parameters", type=["csv"])
                
                if batch_file is not None:
                    try:
                        # Load batch data
                        batch_data = pd.read_csv(batch_file)
                        st.write("Data Preview:")
                        st.dataframe(batch_data.head())
                        
                        # Check if required columns exist
                        if set(app.data.drop('crop_price', axis=1).columns).issubset(set(batch_data.columns)):
                            if st.button("Run Batch Prediction"):
                                # Make predictions
                                predictions = []
                                for _, row in batch_data.iterrows():
                                    success, pred = app.predict_price(row)
                                    if success:
                                        predictions.append(pred)
                                    else:
                                        predictions.append(None)
                                
                                # Add predictions to data
                                batch_data['predicted_price'] = predictions
                                
                                # Show results
                                st.write("Prediction Results:")
                                st.dataframe(batch_data)
                                
                                # Download option
                                st.download_button(
                                    "Download Results as CSV",
                                    batch_data.to_csv(index=False).encode('utf-8'),
                                    "crop_predictions.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                                
                                # Visualize batch predictions
                                st.subheader("Batch Prediction Analysis")
                                
                                # Prediction distribution
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(batch_data['predicted_price'], kde=True, ax=ax)
                                plt.title('Distribution of Predicted Prices')
                                plt.tight_layout()
                                st.pyplot(fig)
                        else:
                            st.error("The uploaded CSV doesn't contain all required columns for prediction.")
                    except Exception as e:
                        st.error(f"Error processing batch file: {str(e)}")

    # Market Analysis Page
    elif page == "Market Analysis":
        st.markdown('<p class="medium-font">Market Price Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        This section provides analysis of market prices and trends for different crops across locations.
        </div>
        """, unsafe_allow_html=True)
        
        # Crop selection
        crop_options = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
        selected_crop = st.selectbox("Select Crop", crop_options)
        
        # Location selection
        location_options = ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Karnataka"]
        selected_location = st.selectbox("Select Location", location_options)
        
        # Time period
        time_period = st.selectbox("Select Time Period", ["Last Week", "Last Month", "Last 3 Months", "Last Year"])
        
        # Generate simulated data for demonstration
        if st.button("Analyze Market Trends"):
            st.write(f"Analyzing {selected_crop} prices in {selected_location} for {time_period.lower()}...")
            
            # Create simulated time series data
            np.random.seed(42)
            
            # Set date range based on selection
            if time_period == "Last Week":
                dates = pd.date_range(end=datetime.now(), periods=7)
            elif time_period == "Last Month":
                dates = pd.date_range(end=datetime.now(), periods=30)
            elif time_period == "Last 3 Months":
                dates = pd.date_range(end=datetime.now(), periods=90)
            else:  # Last Year
                dates = pd.date_range(end=datetime.now(), periods=365)
            
            # Generate price data with trend and seasonality
            base_price = 40  # Base price for rice
            if selected_crop == "Wheat":
                base_price = 30
            elif selected_crop == "Maize":
                base_price = 20
            elif selected_crop == "Cotton":
                base_price = 60
            elif selected_crop == "Sugarcane":
                base_price = 25
            
            # Add location factor
            location_factor = {
                "Punjab": 1.1,
                "Haryana": 1.05,
                "Uttar Pradesh": 0.95,
                "Maharashtra": 1.0,
                "Karnataka": 0.9
            }
            
            # Generate time series with trend and seasonality
            trend = np.linspace(0, 5, len(dates)) if len(dates) > 30 else np.zeros(len(dates))
            seasonality = 2 * np.sin(np.linspace(0, 2 * np.pi, len(dates)))
            noise = np.random.normal(0, 1, len(dates))
            
            prices = base_price * location_factor[selected_location] + trend + seasonality + noise
            
            # Create DataFrame
            market_data = pd.DataFrame({
                'Date': dates,
                'Price': prices
            })
            
            # Show price trend
            st.subheader(f"{selected_crop} Price Trend in {selected_location}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(market_data['Date'], market_data['Price'], 'b-', linewidth=2)
            ax.set_title(f"{selected_crop} Price Trend in {selected_location}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹ per kg)')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show statistics
            st.subheader("Price Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{market_data['Price'].iloc[-1]:.2f}")
            col2.metric("Average Price", f"₹{market_data['Price'].mean():.2f}")
            col3.metric("Min Price", f"₹{market_data['Price'].min():.2f}")
            col4.metric("Max Price", f"₹{market_data['Price'].max():.2f}")
            
            # Calculate price change
            price_change = market_data['Price'].iloc[-1] - market_data['Price'].iloc[0]
            price_change_percent = (price_change / market_data['Price'].iloc[0]) * 100
            
            # Show price change with indicator
            st.metric(
                "Price Change", 
                f"₹{price_change:.2f}", 
                f"{price_change_percent:.2f}%",
                delta_color="normal" if price_change >= 0 else "inverse"
            )
            
            # Show moving average
            st.subheader("Price Trend Analysis")
            window = min(7, len(market_data) // 3)
            market_data['MA'] = market_data['Price'].rolling(window=window).mean()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(market_data['Date'], market_data['Price'], 'b-', label='Daily Price')
            ax.plot(market_data['Date'], market_data['MA'], 'r-', label=f'{window}-Day Moving Average')
            ax.set_title(f"{selected_crop} Price Trend with Moving Average")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹ per kg)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Price volatility
            st.subheader("Price Volatility")
            # Calculate daily returns
            if len(market_data) > 1:
                market_data['Return'] = market_data['Price'].pct_change() * 100
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(market_data['Date'][1:], market_data['Return'][1:], 'g-')
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                ax.set_title(f"{selected_crop} Daily Price Changes")
                ax.set_xlabel('Date')
                ax.set_ylabel('Daily Change (%)')
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Volatility metrics
                volatility = market_data['Return'][1:].std()
                st.write(f"**Price Volatility (Standard Deviation of Returns):** {volatility:.2f}%")
                
                if volatility < 1:
                    st.write("🟢 Low Volatility: Prices are relatively stable.")
                elif volatility < 3:
                    st.write("🟡 Moderate Volatility: Some price fluctuations, but generally predictable.")
                else:
                    st.write("🔴 High Volatility: Significant price fluctuations, market conditions unpredictable.")
            
            # Price forecast (simple)
            st.subheader("Price Forecast (Next 7 Days)")
            
            # Generate forecast using simple linear regression
            from sklearn.linear_model import LinearRegression
            
            # Prepare data
            X = np.array(range(len(market_data))).reshape(-1, 1)
            y = market_data['Price'].values
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make future predictions
            future_X = np.array(range(len(market_data), len(market_data) + 7)).reshape(-1, 1)
            future_dates = pd.date_range(start=market_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
            future_prices = model.predict(future_X)
            
            # Create future dataframe
            future_data = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': future_prices
            })
            
            # Plot actual + forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(market_data['Date'], market_data['Price'], 'b-', label='Historical Price')
            ax.plot(future_data['Date'], future_data['Predicted_Price'], 'r--', label='Forecasted Price')
            ax.set_title(f"{selected_crop} Price Forecast")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹ per kg)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Market recommendations
            st.subheader("Market Recommendations")
            
            last_price = market_data['Price'].iloc[-1]
            forecast_price = future_data['Predicted_Price'].iloc[-1]
            forecast_change = (forecast_price - last_price) / last_price * 100
            
            if forecast_change > 5:
                st.markdown("""
                <div class="success-box">
                <h3>Hold Recommendation</h3>
                <p>Prices are expected to rise in the coming days. Consider holding your crop for better returns.</p>
                </div>
                """, unsafe_allow_html=True)
            elif forecast_change < -5:
                st.markdown("""
                <div class="warning-box">
                <h3>Sell Recommendation</h3>
                <p>Prices are expected to decline in the coming days. Consider selling your crop soon.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                <h3>Neutral Market</h3>
                <p>Prices are expected to remain stable in the coming days. No immediate action required.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Compare with other locations
            st.subheader("Price Comparison Across Locations")
            
            # Generate simulated data for comparison
            comparison_data = []
            for location in location_options:
                loc_factor = location_factor[location]
                current_price = market_data['Price'].iloc[-1] * loc_factor / location_factor[selected_location]
                comparison_data.append({
                    'Location': location,
                    'Current Price': current_price
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Bar chart visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(comparison_df['Location'], comparison_df['Current Price'], color='skyblue')
            
            # Highlight selected location
            selected_idx = comparison_df[comparison_df['Location'] == selected_location].index[0]
            bars[selected_idx].set_color('orange')
            
            ax.set_title(f"{selected_crop} Price Comparison Across Locations")
            ax.set_xlabel('Location')
            ax.set_ylabel('Price (₹ per kg)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'₹{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)

    # About Page
    elif page == "About":
        st.markdown('<p class="medium-font">About This Application</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Advanced Crop Price Prediction System</h3>
        <p>This application is designed to help farmers and agricultural stakeholders make informed decisions 
        by predicting crop prices based on various factors and comparing them with real-time market rates.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Features")
        st.markdown("""
        - **Data Management**: Upload and analyze crop data
        - **Model Training**: Train machine learning models for price prediction
        - **Price Prediction**: Predict crop prices based on multiple factors
        - **Market Analysis**: Analyze real-time market trends and forecasts
        - **Batch Processing**: Process multiple predictions at once
        """)
        
        st.subheader("How It Works")
        st.markdown("""
        1. **Data Collection**: The system uses CSV data containing weather quality, soil conditions, crop types, and other relevant parameters.
        2. **Model Training**: It employs Random Forest regression to learn patterns from historical data.
        3. **Prediction**: Based on the trained model, it predicts crop prices for given conditions.
        4. **Market Integration**: It compares predicted prices with real-time market rates.
        5. **Analysis**: It provides insights and recommendations based on the prediction and market data.
        """)
        
        st.subheader("Required Data Format")
        st.markdown("""
        The CSV file should contain the following types of columns:
        - **Weather parameters**: Temperature, rainfall, humidity, etc.
        - **Soil parameters**: pH, nutrient content, soil type, etc.
        - **Crop information**: Crop type, variety, growth stage, etc.
        - **Other factors**: Irrigation method, fertilizer application, etc.
        - **Target variable**: 'crop_price' - the actual price to train the model
        """)
        
        # Sample CSV structure
        sample_data = {
            'soil_ph': [6.5, 7.2, 5.8, 6.9, 7.0],
            'soil_nutrient': ['high', 'medium', 'low', 'high', 'medium'],
            'temperature': [32.5, 28.4, 30.2, 29.8, 31.5],
            'rainfall': [120, 85, 150, 95, 110],
            'humidity': [65, 72, 68, 70, 75],
            'crop_type': ['rice', 'wheat', 'rice', 'maize', 'wheat'],
            'irrigation': ['flood', 'drip', 'flood', 'sprinkler', 'drip'],
            'crop_price': [45.2, 32.1, 44.8, 25.6, 33.5]
        }
        
        st.write("Sample CSV structure:")
        st.dataframe(pd.DataFrame(sample_data))
        
        # Download sample CSV
        sample_df = pd.DataFrame(sample_data)
        st.download_button(
            "Download Sample CSV",
            sample_df.to_csv(index=False).encode('utf-8'),
            "sample_crop_data.csv",
            "text/csv",
            key='download-sample-csv'
        )
        
        st.subheader("Technologies Used")
        st.markdown("""
        - **Python**: Core programming language
        - **Pandas & NumPy**: Data manipulation and numerical computing
        - **Scikit-learn**: Machine learning algorithms
        - **Matplotlib & Seaborn**: Data visualization
        - **Streamlit**: Web application framework
        - **Joblib**: Model serialization
        """)
        
        st.subheader("Future Enhancements")
        st.markdown("""
        - **Weather API Integration**: Real-time weather data for better predictions
        - **Advanced Models**: Implementation of deep learning models for higher accuracy
        - **Mobile Application**: Cross-platform mobile app for field use
        - **Crop Recommendation**: Recommend optimal crops based on conditions
        - **Supply Chain Integration**: Connect with supply chain logistics
        """)


def run_command_line_app():
    """Command line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crop Price Prediction CLI')
    parser.add_argument('--input', '-i', help='Input CSV file path', required=True)
    parser.add_argument('--model', '-m', help='Model file path', required=True)
    parser.add_argument('--output', '-o', help='Output CSV file path', default='predictions.csv')
    parser.add_argument('--market', action='store_true', help='Compare with market prices')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    app = CropPredictionApp()
    success, message = app.load_model(args.model)
    
    if not success:
        print(f"Error: {message}")
        return
    
    print(f"Loading data from {args.input}...")
    data = pd.read_csv(args.input)
    
    print("Making predictions...")
    predictions = []
    for _, row in data.iterrows():
        success, pred = app.predict_price(row)
        if success:
            predictions.append(pred)
        else:
            predictions.append(None)
    
    data['predicted_price'] = predictions
    
    if args.market:
        print("Fetching market prices...")
        market_prices = []
        for _, row in data.iterrows():
            if 'crop_type' in row and 'location' in row:
                success, market_data = app.get_market_price(row['crop_type'], row['location'])
                if success:
                    market_prices.append(market_data['current_price'])
                else:
                    market_prices.append(None)
            else:
                market_prices.append(None)
        
        data['market_price'] = market_prices
        
        # Calculate difference
        mask = (~data['predicted_price'].isna()) & (~data['market_price'].isna())
        data.loc[mask, 'price_difference'] = data.loc[mask, 'predicted_price'] - data.loc[mask, 'market_price']
        data.loc[mask, 'price_difference_percent'] = (data.loc[mask, 'price_difference'] / data.loc[mask, 'market_price']) * 100
    
    print(f"Saving results to {args.output}...")
    data.to_csv(args.output, index=False)
    print("Done!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        run_command_line_app()
    else:
        # Web app mode
        create_streamlit_app()