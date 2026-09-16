import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Foamed Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .best-model-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        transition: transform 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        color: white;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏗️ Foamed Concrete Strength Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict compressive strength using advanced machine learning models trained on experimental data</div>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models and related files"""
    models = {}
    scaler = None
    feature_names = None
    model_metrics = None
    
    try:
        # Load scaler
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
        
        # Load feature names
        if os.path.exists('feature_names.pkl'):
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        
        # Load model metrics
        if os.path.exists('model_metrics.pkl'):
            with open('model_metrics.pkl', 'rb') as f:
                model_metrics = pickle.load(f)
        
        # Load XGBoost (BEST MODEL)
        if os.path.exists('xgboost_model.json'):
            import xgboost as xgb
            model_xgb = xgb.XGBRegressor()
            model_xgb.load_model('xgboost_model.json')
            models['XGBoost ⭐'] = {
                'model': model_xgb,
                'is_best': True,
                'description': 'Extreme Gradient Boosting - Best performing model'
            }
        
        # Load CatBoost
        if os.path.exists('catboost_model.cbm'):
            from catboost import CatBoostRegressor
            model_cat = CatBoostRegressor()
            model_cat.load_model('catboost_model.cbm')
            models['CatBoost'] = {
                'model': model_cat,
                'is_best': False,
                'description': 'Categorical Boosting - Robust performance'
            }
        
        # Load AdaBoost
        if os.path.exists('adaboost_model.pkl'):
            model_ada = joblib.load('adaboost_model.pkl')
            models['AdaBoost'] = {
                'model': model_ada,
                'is_best': False,
                'description': 'Adaptive Boosting - Simple ensemble method'
            }
        
        if not models:
            st.error("❌ No models found! Please ensure model files exist.")
            
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
    
    return models, scaler, feature_names, model_metrics

# Load models
with st.spinner('Loading models...'):
    models, scaler, feature_names, model_metrics = load_models()

if models:
    # Sidebar inputs
    st.sidebar.markdown("## 📊 Mix Design Parameters")
    st.sidebar.markdown("---")
    
    with st.sidebar:
        # Input fields
        col1, col2 = st.columns(2)
        with col1:
            cement = st.number_input(
                'Cement (C) (kg/m³)', 
                min_value=0.0, 
                max_value=2000.0, 
                value=500.0,
                step=10.0,
                help="Amount of cement in kg per cubic meter"
            )
        with col2:
            scm = st.number_input(
                'SCM (kg/m³)', 
                min_value=0.0, 
                max_value=1000.0, 
                value=0.0,
                step=10.0,
                help="Supplementary Cementitious Materials"
            )
        
        sand = st.number_input(
            'Fine Aggregate/Sand (S) (kg/m³)', 
            min_value=0.0, 
            max_value=2000.0, 
            value=800.0,
            step=10.0,
            help="Amount of fine aggregate"
        )
        
        water = st.number_input(
            'Water (W) (kg/m³)', 
            min_value=0.0, 
            max_value=1000.0, 
            value=250.0,
            step=5.0,
            help="Amount of water"
        )
        
        foam = st.number_input(
            'Foam (F) (kg/m³)', 
            min_value=0.0, 
            max_value=500.0, 
            value=40.0,
            step=1.0,
            help="Amount of foam"
        )
        
        target_density = st.number_input(
            'Target Density (kg/m³)', 
            min_value=300.0, 
            max_value=2500.0, 
            value=1800.0,
            step=50.0,
            help="Target density of foamed concrete"
        )
        
        curing_time = st.number_input(
            'Curing Time (days)', 
            min_value=1, 
            max_value=365, 
            value=28,
            step=1,
            help="Number of days cured"
        )
        
        st.markdown("---")
        
        # Model selection
        st.markdown("### 🤖 Select Model")
        model_names = list(models.keys())
        default_index = 0
        for i, name in enumerate(model_names):
            if models[name]['is_best']:
                default_index = i
                break
        
        selected_model = st.selectbox(
            'Choose Model',
            model_names,
            index=default_index,
            help="XGBoost is the best performing model (⭐)"
        )
        
        if models[selected_model]['is_best']:
            st.markdown('<span class="best-model-badge">⭐ BEST PERFORMING MODEL</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        predict_button = st.button("🔮 Predict Strength", use_container_width=True)
    
    # Main area - input summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Binder", f"{cement + scm:.0f} kg/m³")
    with col2:
        w_b_ratio = water/(cement+scm+1e-10)
        st.metric("Water/Binder Ratio", f"{w_b_ratio:.3f}")
    with col3:
        f_b_ratio = foam/(cement+scm+1e-10)
        st.metric("Foam/Binder Ratio", f"{f_b_ratio:.3f}")
    with col4:
        st.metric("Target Density", f"{target_density:.0f} kg/m³")
    
    # Display model metrics in sidebar
    if model_metrics:
        with st.sidebar.expander("📊 Model Performance Metrics", expanded=False):
            for model_name, metrics in model_metrics.items():
                display_name = model_name + " ⭐" if model_name == 'XGBoost' else model_name
                st.markdown(f"**{display_name}**")
                st.write(f"R²: {metrics.get('R2', 'N/A'):.4f}")
                st.write(f"RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
                st.write(f"MAE: {metrics.get('MAE', 'N/A'):.4f}")
                st.markdown("---")
    
    # Prediction logic
    if predict_button:
        try:
            # Calculate derived features
            total_binder = cement + scm
            cement_ratio = cement / (total_binder + sand + 1e-10)
            scm_ratio = scm / (total_binder + sand + 1e-10)
            sand_ratio = sand / (total_binder + sand + 1e-10)
            water_to_binder = water / (total_binder + 1e-10)
            foam_to_binder = foam / (total_binder + 1e-10)
            sand_to_binder = sand / (total_binder + 1e-10)
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Cement (C)': [cement],
                'SCM': [scm],
                'F/A Sand (S)': [sand],
                'Water (W)': [water],
                'Foam (F)': [foam],
                'Target Density (kg/m3)': [target_density],
                'Curing Time (days)': [curing_time],
                'Cement_Ratio': [cement_ratio],
                'SCM_Ratio': [scm_ratio],
                'Sand_Ratio': [sand_ratio],
                'Total_Binder': [total_binder],
                'Water_to_Binder': [water_to_binder],
                'Foam_to_Binder': [foam_to_binder],
                'Sand_to_Binder': [sand_to_binder]
            })
            
            # Scale input
            if scaler is not None:
                input_scaled = scaler.transform(input_data)
            else:
                input_scaled = input_data.values
            
            # Make prediction
            selected_model_obj = models[selected_model]['model']
            model_name_clean = selected_model.replace(' ⭐', '')
            
            if model_name_clean == 'CatBoost':
                prediction = selected_model_obj.predict(input_scaled)[0]
            else:
                prediction = selected_model_obj.predict(input_scaled)[0]
            
            # Display prediction
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<div class="prediction-label">Predicted Compressive Strength</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-value">{prediction:.2f} MPa</div>', unsafe_allow_html=True)
                
                # Determine strength category
                if prediction < 5:
                    category = "Very Low Strength"
                    color = "#ff6b6b"
                    emoji = "🔴"
                elif prediction < 10:
                    category = "Low Strength"
                    color = "#ffa94d"
                    emoji = "🟠"
                elif prediction < 20:
                    category = "Medium Strength"
                    color = "#ffd93d"
                    emoji = "🟡"
                elif prediction < 35:
                    category = "High Strength"
                    color = "#6bcb77"
                    emoji = "🟢"
                else:
                    category = "Very High Strength"
                    color = "#4d96ff"
                    emoji = "🔵"
                
                st.markdown(f'<div style="color: white; font-size: 1.3rem; font-weight: bold; margin-top: 0.5rem;">{emoji} {category}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="color: white; font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Using: {selected_model}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Get predictions from all models for comparison
            st.markdown("### 📊 Model Comparison")
            
            all_predictions = {}
            for name, model_info in models.items():
                try:
                    model = model_info['model']
                    model_name_clean = name.replace(' ⭐', '')
                    
                    if scaler is not None:
                        input_scaled = scaler.transform(input_data)
                    else:
                        input_scaled = input_data.values
                    
                    if model_name_clean == 'CatBoost':
                        pred = model.predict(input_scaled)[0]
                    else:
                        pred = model.predict(input_scaled)[0]
                    all_predictions[name] = pred
                except Exception as e:
                    all_predictions[name] = None
            
            # Create comparison chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Model Predictions", "Prediction Distribution"),
                specs=[[{"type": "bar"}, {"type": "box"}]]
            )
            
            # Bar chart
            model_names_list = list(all_predictions.keys())
            pred_values = [all_predictions[name] for name in model_names_list if all_predictions[name] is not None]
            valid_names = [name for name in model_names_list if all_predictions[name] is not None]
            colors = ['#ff6b6b' if name == selected_model else '#1f77b4' for name in valid_names]
            
            fig.add_trace(
                go.Bar(
                    x=valid_names,
                    y=pred_values,
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in pred_values],
                    textposition='outside',
                    name='Predictions'
                ),
                row=1, col=1
            )
            
            # Add horizontal line for selected prediction
            fig.add_hline(
                y=prediction,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Selected: {prediction:.2f} MPa",
                row=1, col=1
            )
            
            # Box plot for distribution
            fig.add_trace(
                go.Box(
                    y=pred_values,
                    name='All Predictions',
                    boxmean='sd',
                    marker_color='#1f77b4'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                template="plotly_white"
            )
            
            fig.update_xaxes(title_text="Model", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Strength (MPa)", row=1, col=1)
            fig.update_yaxes(title_text="Strength (MPa)", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display model metrics comparison
            if model_metrics:
                st.markdown("### 📈 Model Performance Comparison")
                
                # Create metrics DataFrame
                metrics_df = pd.DataFrame(model_metrics).T
                metrics_df = metrics_df.round(4)
                
                # Add prediction column
                metrics_df['Prediction (MPa)'] = [all_predictions.get(name + ' ⭐' if name == 'XGBoost' else name, None) 
                                                   for name in metrics_df.index]
                
                # Style the dataframe
                st.dataframe(
                    metrics_df.style.background_gradient(cmap='Blues', axis=0),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all required model files are present.")
    
    # Information section
    with st.expander("ℹ️ About the Models", expanded=False):
        st.markdown("""
        ### Models Used
        
        | Model | Description | Status |
        |-------|-------------|--------|
        | **XGBoost** ⭐ | Extreme Gradient Boosting - Best performing model with excellent accuracy and speed | **BEST MODEL** |
        | **CatBoost** | Categorical Boosting - Handles categorical features well, robust performance | Available |
        | **AdaBoost** | Adaptive Boosting - Simple but effective ensemble method | Available |
        
        ### Why XGBoost is the Best?
        - Superior handling of complex relationships
        - Built-in regularization to prevent overfitting
        - Efficient and scalable
        - Handles missing values automatically
        - Best R² score and lowest error rates
        
        ### Dataset Information
        - **Source**: Experimental data on foamed concrete
        - **Mix Design Parameters**: Cement, SCM, Sand, Water, Foam, Target Density, Curing Time
        - **Target**: Compressive Strength (MPa)
        - **Samples**: 200+ mix designs
        
        ### How to Use
        1. Enter the mix design parameters in the sidebar
        2. Select a model (XGBoost is recommended as it's the best)
        3. Click "Predict Strength" to get results
        4. Compare predictions from different models
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏗️ Foamed Concrete Strength Predictor | Made with ❤️ using Streamlit</p>
    <p style='font-size: 0.8rem;'>Powered by XGBoost, CatBoost, and AdaBoost | Best Model: XGBoost ⭐</p>
    <p style='font-size: 0.7rem; color: #999;'>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Error message if no models are loaded
    st.error("""
    ### ⚠️ No Models Found
    
    Please ensure that the following model files are in the same directory as `app.py`:
    
    **Required files:**
    - `scaler.pkl`
    - `feature_names.pkl`
    - `xgboost_model.json` ⭐ **(Your best model)**
    
    **Optional files:**
    - `catboost_model.cbm`
    - `adaboost_model.pkl`
    - `model_metrics.pkl`
    
    **To fix this:**
    1. Make sure you've saved your models from Kaggle
    2. Upload all model files to this Space
    3. Check the file names match exactly
    """)
    
    # Show file upload option
    st.markdown("### 📤 Upload Model Files")
    st.info("""
    If you have the model files, you can upload them here:
    - Click on the 'Files' tab in your Hugging Face Space
    - Click 'Add file' → 'Upload files'
    - Select all your model files
    - Commit changes
    """)