import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Import XGBoost or use fallback
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    XGB_AVAILABLE = False
    st.warning("XGBoost not installed. Using GradientBoostingClassifier as fallback.")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Hypertension Risk Predictor Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498db;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin-bottom: 1rem;
    }
    .stNumberInput > div > div > input {
        background-color: #f0f2f6;
    }
    .upload-box {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🫀 Hypertension Risk Predictor Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive dashboard analyzes key lifestyle and environmental predictors of hypertension.
Upload your dataset to explore the data, understand risk factors, and see how different variables influence hypertension risk.
""")

# Initialize session state for data storage
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'X_test_scaled' not in st.session_state:
    st.session_state.X_test_scaled = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_encoded' not in st.session_state:
    st.session_state.X_encoded = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

def validate_dataset(df):
    """Validate the uploaded dataset has required columns and valid data"""
    required_columns = [
        'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI', 
        'Family_History', 'Exercise_Level', 'Smoking_Status', 'Has_Hypertension'
    ]
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check data types
    try:
        # Convert to appropriate types
        df['Age'] = pd.to_numeric(df['Age'])
        df['Salt_Intake'] = pd.to_numeric(df['Salt_Intake'])
        df['Stress_Score'] = pd.to_numeric(df['Stress_Score'])
        df['Sleep_Duration'] = pd.to_numeric(df['Sleep_Duration'])
        df['BMI'] = pd.to_numeric(df['BMI'])
        
        # Check for valid values
        if df['Age'].min() < 0 or df['Age'].max() > 120:
            return False, "Age values should be between 0 and 120"
        
        if df['Stress_Score'].min() < 0 or df['Stress_Score'].max() > 10:
            return False, "Stress_Score should be between 0 and 10"
            
        if df['BMI'].min() < 10 or df['BMI'].max() > 60:
            return False, "BMI values appear unrealistic (should be between 10-60)"
            
    except Exception as e:
        return False, f"Data type error: {str(e)}"
    
    return True, "Dataset validated successfully"

def preprocess_data(df):
    """Preprocess the dataset"""
    df_processed = df.copy()
    
    # Ensure proper data types
    df_processed['Age'] = pd.to_numeric(df_processed['Age'], errors='coerce')
    df_processed['Salt_Intake'] = pd.to_numeric(df_processed['Salt_Intake'], errors='coerce')
    df_processed['Stress_Score'] = pd.to_numeric(df_processed['Stress_Score'], errors='coerce')
    df_processed['Sleep_Duration'] = pd.to_numeric(df_processed['Sleep_Duration'], errors='coerce')
    df_processed['BMI'] = pd.to_numeric(df_processed['BMI'], errors='coerce')
    
    # Create age groups (as strings to avoid Interval objects)
    bins = [0, 18, 25, 35, 45, 55, 65, 85, 120]
    labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-84', '85+']
    df_processed['Age_Group'] = pd.cut(df_processed['Age'], bins=bins, labels=labels, right=False)
    df_processed['Age_Group'] = df_processed['Age_Group'].astype(str)  # Convert to string
    
    # Create BMI categories (as strings to avoid Interval objects)
    df_processed['BMI_Category'] = pd.cut(df_processed['BMI'], 
                                        bins=[0, 18.5, 25, 30, 100],
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df_processed['BMI_Category'] = df_processed['BMI_Category'].astype(str)  # Convert to string
    
    # Clean categorical variables
    categorical_cols = ['Family_History', 'Exercise_Level', 'Smoking_Status', 'Has_Hypertension']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip()
    
    return df_processed

@st.cache_resource
def train_models_improved(df):
    """Train machine learning models for hypertension prediction using reference code approach"""
    try:
        # 1. Filter data for young adults (18-35) like reference code
        df_filtered = df[(df['Age'] >= 18) & (df['Age'] <= 35)].copy()
        
        if len(df_filtered) < 50:
            st.warning(f"Only {len(df_filtered)} samples in age range 18-35. Using full dataset instead.")
            df_filtered = df.copy()
        
        # 2. Feature Selection & Preprocessing
        target = 'Has_Hypertension'
        
        # We drop metadata/target, but REMOVE 'BP_History' and 'Medication' as requested
        cols_to_drop = [target, 'Age_Group', 'Age_Category', 'Patient_ID', 'BMI_Category']
        # Only drop columns that exist
        features_to_drop = [c for c in cols_to_drop if c in df_filtered.columns]
        
        X = df_filtered.drop(columns=features_to_drop)
        y = df_filtered[target].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        
        # Check if y has both classes
        if y.nunique() < 2:
            raise ValueError("Target variable 'Has_Hypertension' must have both 'Yes' and 'No' values")
        
        # ENCODING: Using get_dummies instead of LabelEncoder for better interpretability
        X_encoded = pd.get_dummies(X, drop_first=False)
        feature_names = X_encoded.columns.tolist()
        
        # Split the data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling: Required for Logistic Regression performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Model Development - Using XGBoost instead of Gradient Boosting
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(random_state=42)
        }
        
        trained_models = {}
        results = {}
        feature_importances = {}
        
        for name, model in models.items():
            try:
                # Scale data for Logistic Regression, use raw data for Trees
                if name == "Logistic Regression":
                    X_tr = X_train_scaled
                    X_ts = X_test_scaled
                else:
                    X_tr = X_train.values if hasattr(X_train, 'values') else X_train
                    X_ts = X_test.values if hasattr(X_test, 'values') else X_test
                
                model.fit(X_tr, y_train)
                trained_models[name] = model
                
                y_pred = model.predict(X_ts)
                y_prob = model.predict_proba(X_ts)[:, 1]
                
                auc = roc_auc_score(y_test, y_prob)
                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                results[name] = {
                    'Accuracy': acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'AUC-ROC': auc
                }
                
                # Get feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    feature_importances[name] = dict(zip(feature_names, model.feature_importances_))
                
                # Get coefficients for Logistic Regression
                if name == "Logistic Regression":
                    feature_importances[name + '_Coefficients'] = dict(zip(feature_names, model.coef_[0]))
                    
            except Exception as e:
                st.warning(f"Could not train {name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No models could be trained successfully")
        
        # Store additional data in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.scaler = scaler
        st.session_state.trained_models = trained_models
        st.session_state.X_encoded = X_encoded
        st.session_state.feature_names = feature_names
        
        return results, feature_importances, trained_models
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None

def get_model_coefficients():
    """Extract model coefficients for visualization"""
    if not st.session_state.trained_models:
        return None
    
    coefficients = {}
    
    # Logistic Regression Coefficients
    if "Logistic Regression" in st.session_state.trained_models:
        log_model = st.session_state.trained_models["Logistic Regression"]
        if hasattr(log_model, 'coef_'):
            coefficients['Logistic_Regression'] = pd.Series(
                log_model.coef_[0], 
                index=st.session_state.feature_names
            ).sort_values()
    
    # Random Forest Feature Importance
    if "Random Forest" in st.session_state.trained_models:
        rf_model = st.session_state.trained_models["Random Forest"]
        if hasattr(rf_model, 'feature_importances_'):
            coefficients['Random_Forest'] = pd.Series(
                rf_model.feature_importances_,
                index=st.session_state.feature_names
            ).sort_values(ascending=False)
    
    # XGBoost Feature Importance
    if "XGBoost" in st.session_state.trained_models:
        xgb_model = st.session_state.trained_models["XGBoost"]
        if hasattr(xgb_model, 'feature_importances_'):
            coefficients['XGBoost'] = pd.Series(
                xgb_model.feature_importances_,
                index=st.session_state.feature_names
            ).sort_values(ascending=False)
    
    return coefficients

def calculate_simplified_risk(age, bmi, family_history, exercise_level, smoking_status, salt_intake, stress_score, sleep_duration):
    """Calculate risk using simplified scoring system (removed BP_History and Medication)"""
    risk_score = 0
    
    # Age contribution
    if age < 30:
        risk_score += 10
    elif age < 40:
        risk_score += 20
    elif age < 50:
        risk_score += 35
    elif age < 60:
        risk_score += 50
    else:
        risk_score += 70
    
    # BMI contribution
    if bmi < 18.5:
        risk_score += 5
    elif bmi < 25:
        risk_score += 10
    elif bmi < 30:
        risk_score += 30
    else:
        risk_score += 50
    
    # Family history
    if family_history == "Yes":
        risk_score += 25
    
    # Exercise level
    exercise_scores = {"Low": 25, "Moderate": 10, "High": 0}
    risk_score += exercise_scores[exercise_level]
    
    # Smoking status
    if smoking_status == "Smoker":
        risk_score += 20
    
    # Salt intake
    if salt_intake > 8:
        excess_salt = salt_intake - 8
        risk_score += excess_salt * 3
    
    # Stress score
    risk_score += stress_score * 2
    
    # Sleep duration
    if sleep_duration < 6:
        risk_score += (6 - sleep_duration) * 5
    elif sleep_duration > 9:
        risk_score += (sleep_duration - 9) * 3
    
    # Normalize to 0-100
    risk_score = min(100, max(0, risk_score))
    
    return risk_score

def get_recommendations(risk_level):
    """Get recommendations based on risk level"""
    if risk_level == 'low':
        return [
            "Maintain your healthy lifestyle habits",
            "Continue regular physical activity (at least 150 minutes/week)",
            "Monitor blood pressure annually",
            "Maintain a balanced diet with limited salt intake",
            "Ensure adequate sleep (7-9 hours per night)"
        ]
    elif risk_level == 'moderate':
        return [
            "Consider lifestyle modifications to reduce risk",
            "Aim to reduce salt intake to less than 6g per day",
            "Increase physical activity to moderate intensity",
            "Monitor blood pressure every 6 months",
            "Practice stress management techniques",
            "Maintain healthy weight (BMI 18.5-25)"
        ]
    elif risk_level == 'high':
        return [
            "Consult with a healthcare provider for evaluation",
            "Implement significant lifestyle changes immediately",
            "Consider medical monitoring of blood pressure",
            "Reduce salt intake to less than 5g per day",
            "Increase exercise to at least 30 minutes daily",
            "Practice daily stress reduction techniques",
            "Monitor blood pressure monthly"
        ]
    else:  # very_high
        return [
            "Urgent consultation with healthcare provider recommended",
            "Immediate lifestyle intervention required",
            "Regular medical monitoring essential",
            "Consider medication if lifestyle changes insufficient",
            "Comprehensive cardiovascular assessment needed",
            "Weekly blood pressure monitoring",
            "Consider referral to specialist"
        ]

def display_risk_results(risk_score, risk_category, color, recommendations, 
                        age, bmi, family_history, exercise_level, smoking_status, salt_intake, 
                        stress_score, sleep_duration, ml_based=False):
    """Display risk assessment results"""
    st.markdown("## 📊 Your Risk Assessment Results")
    
    if ml_based:
        st.info("✅ Using ML model trained on your dataset")
    else:
        st.warning("⚠️ Using simplified scoring (train ML models for more accuracy)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {color};">
        <h3>Risk Score</h3>
        <h1 style="color: {color};">{risk_score:.0f}/100</h1>
        <p><strong>Category:</strong> {risk_category}</p>
        <p><strong>Interpretation:</strong> {risk_category.lower()} of developing hypertension</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Hypertension Risk", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "yellow"},
                    {'range': [50, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        
        fig.update_layout(height=300, template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Risk factor summary
        st.markdown("#### 🔍 Your Input Summary")
        
        risk_factors = []
        if age >= 40:
            risk_factors.append(f"Age: {age} (≥40)")
        if bmi >= 25:
            risk_factors.append(f"BMI: {bmi:.1f} (≥25)")
        if family_history == "Yes":
            risk_factors.append("Family history: Positive")
        if exercise_level == "Low":
            risk_factors.append("Exercise: Low")
        if smoking_status == "Smoker":
            risk_factors.append("Smoking: Yes")
        if salt_intake > 8:
            risk_factors.append(f"Salt intake: {salt_intake}g/day (>8g)")
        if stress_score > 5:
            risk_factors.append(f"Stress: {stress_score}/10 (>5)")
        if sleep_duration < 6 or sleep_duration > 9:
            risk_factors.append(f"Sleep: {sleep_duration} hours (not 7-9)")
        
        if risk_factors:
            st.markdown("**Identified Factors:**")
            for factor in risk_factors[:5]:  # Show top 5
                st.markdown(f"• {factor}")
            if len(risk_factors) > 5:
                st.markdown(f"*... and {len(risk_factors)-5} more*")
        else:
            st.markdown("**No major risk factors identified**")
        
        st.markdown("---")
        st.markdown(f"**Total factors:** {len(risk_factors)}")
    
    # Display recommendations
    st.markdown("## 💡 Personalized Recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. **{rec}**")
    
    # Comparative analysis with uploaded dataset
    if st.session_state.df is not None:
        st.markdown("## 📈 How You Compare to the Dataset")
        
        # Calculate statistics from uploaded dataset
        dataset_stats = st.session_state.df['Has_Hypertension'].str.lower() == 'yes'
        dataset_hypertension_rate = dataset_stats.mean() * 100
        
        # Find similar individuals in dataset
        similar_criteria = []
        if 'Age' in st.session_state.df.columns:
            similar_criteria.append((st.session_state.df['Age'] >= age-5) & (st.session_state.df['Age'] <= age+5))
        if 'Exercise_Level' in st.session_state.df.columns:
            similar_criteria.append(st.session_state.df['Exercise_Level'] == exercise_level)
        if 'Smoking_Status' in st.session_state.df.columns:
            similar_criteria.append(st.session_state.df['Smoking_Status'] == smoking_status)
        if 'Family_History' in st.session_state.df.columns:
            similar_criteria.append(st.session_state.df['Family_History'] == family_history)
        
        if similar_criteria:
            combined_criteria = similar_criteria[0]
            for criterion in similar_criteria[1:]:
                combined_criteria = combined_criteria & criterion
            
            similar_individuals = st.session_state.df[combined_criteria]
            
            if len(similar_individuals) > 0:
                similar_hypertension_rate = (similar_individuals['Has_Hypertension'].str.lower() == 'yes').mean() * 100
                
                comparison_df = pd.DataFrame({
                    'Group': ['Similar Individuals in Dataset', 'Overall Dataset'],
                    'Hypertension Rate (%)': [similar_hypertension_rate, dataset_hypertension_rate]
                })
                
                fig = px.bar(comparison_df, x='Group', y='Hypertension Rate (%)',
                            title='Comparison with Dataset',
                            color='Group',
                            color_discrete_sequence=['#3498db', '#2ecc71'],
                            text='Hypertension Rate (%)')
                fig.update_layout(template=chart_theme)
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
    
    # Download report
    st.markdown("---")
    st.markdown("### 📥 Download Your Report")
    
    report_content = f"""
    HYPERTENSION RISK ASSESSMENT REPORT
    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    {'Using ML Model: YES' if ml_based else 'Using Simplified Scoring'}
    
    PERSONAL INFORMATION:
    - Age: {age} years
    - BMI: {bmi:.1f}
    - Family History: {family_history}
    - Exercise Level: {exercise_level}
    - Smoking Status: {smoking_status}
    - Salt Intake: {salt_intake}g/day
    - Stress Level: {stress_score}/10
    - Sleep Duration: {sleep_duration} hours
    
    RISK ASSESSMENT:
    - Risk Score: {risk_score:.0f}/100
    - Risk Category: {risk_category}
    
    RECOMMENDATIONS:
    {chr(10).join([f'{i}. {rec}' for i, rec in enumerate(recommendations, 1)])}
    
    IMPORTANT NOTES:
    - This assessment is for informational purposes only
    - Not a substitute for professional medical advice
    - Consult with healthcare provider for personalized guidance
    - Regular monitoring is essential for hypertension management
    
    Based on analysis of uploaded dataset: {len(st.session_state.df) if st.session_state.df is not None else 0} records
    Dataset hypertension rate: {dataset_hypertension_rate:.1f}%
    
    Stay healthy and proactive about your cardiovascular health!
    """
    
    st.download_button(
        label="Download Full Report",
        data=report_content,
        file_name=f"hypertension_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Sidebar for navigation and controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.markdown("## Navigation")
    
    page = st.radio(
        "Select a page:",
        ["📁 Upload Data", "📊 Dataset Overview", "🔍 Exploratory Analysis", "📈 Predictive Models", "🎯 Risk Assessment", "📋 About"]
    )
    
    # Data filters (only show if data is loaded)
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### Data Filters")
        
        # Age filter
        age_min = int(st.session_state.df['Age'].min())
        age_max = int(st.session_state.df['Age'].max())
        age_range = st.slider(
            "Select Age Range:",
            min_value=age_min,
            max_value=age_max,
            value=(max(18, age_min), min(35, age_max)),
            help="Filter data by age range"
        )
        
        # Hypertension status filter
        hypertension_filter = st.multiselect(
            "Hypertension Status:",
            options=["All", "Hypertensive", "Non-Hypertensive"],
            default=["All"]
        )
        
        st.markdown("---")
        st.markdown("### Visualization Settings")
        
        chart_theme = st.selectbox(
            "Chart Theme:",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
        )
    
    st.markdown("---")
    st.markdown("#### About")
    st.markdown("Upload a CSV file with hypertension data to begin analysis.")

# Apply filters function
def apply_filters(df, age_range, hypertension_filter):
    """Apply filters to the dataset"""
    if df is None:
        return None
    
    df_filtered = df.copy()
    
    # Age filter
    df_filtered = df_filtered[(df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1])]
    
    # Hypertension filter
    if "All" not in hypertension_filter:
        hypertension_labels = {'Hypertensive': 'Yes', 'Non-Hypertensive': 'No'}
        selected_status = [hypertension_labels[s] for s in hypertension_filter]
        df_filtered = df_filtered[df_filtered['Has_Hypertension'].isin(selected_status)]
    
    return df_filtered

# Apply filters if data exists
if st.session_state.df is not None and 'age_range' in locals():
    st.session_state.df_filtered = apply_filters(
        st.session_state.df, 
        age_range, 
        hypertension_filter
    )
else:
    st.session_state.df_filtered = st.session_state.df

# Page routing
if page == "📁 Upload Data":
    st.markdown('<h2 class="sub-header">Upload Your Hypertension Dataset</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Dataset Requirements:</strong>
    <ul>
    <li>CSV file format</li>
    <li>Must contain these 9 columns (exact names):</li>
    </ul>
    <ol>
    <li><code>Age</code> (numerical)</li>
    <li><code>Salt_Intake</code> (numerical)</li>
    <li><code>Stress_Score</code> (numerical, 0-10)</li>
    <li><code>Sleep_Duration</code> (numerical)</li>
    <li><code>BMI</code> (numerical)</li>
    <li><code>Family_History</code> (categorical: Yes/No)</li>
    <li><code>Exercise_Level</code> (categorical: Low/Moderate/High)</li>
    <li><code>Smoking_Status</code> (categorical: Smoker/Non-Smoker)</li>
    <li><code>Has_Hypertension</code> (categorical: Yes/No)</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate the dataset
            is_valid, message = validate_dataset(df)
            
            if is_valid:
                # Preprocess the data
                df_processed = preprocess_data(df)
                
                # Store in session state
                st.session_state.df = df_processed
                st.session_state.df_filtered = df_processed
                st.session_state.models_trained = False
                st.session_state.model_results = None
                st.session_state.feature_importances = None
                st.session_state.best_model = None
                st.session_state.trained_models = None
                
                st.success(f"✅ {message}")
                st.success(f"📊 Dataset loaded successfully: {len(df_processed)} records")
                
                # Show dataset preview
                with st.expander("Preview Dataset", expanded=True):
                    st.dataframe(df_processed.head(10), use_container_width=True)
                    
                    # Basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df_processed))
                    with col2:
                        hypertension_rate = (df_processed['Has_Hypertension'].str.lower() == 'yes').mean() * 100
                        st.metric("Hypertension Rate", f"{hypertension_rate:.1f}%")
                    with col3:
                        avg_age = df_processed['Age'].mean()
                        st.metric("Average Age", f"{avg_age:.1f}")
                
                # Train models button
                if st.button("🚀 Train Machine Learning Models (Improved)", type="primary"):
                    with st.spinner("Training models using improved method... This may take a few seconds."):
                        result = train_models_improved(df_processed)
                        
                        if result is not None:
                            st.session_state.model_results, st.session_state.feature_importances, trained_models = result
                            st.session_state.models_trained = True
                            
                            # Find best model
                            if st.session_state.model_results:
                                results_df = pd.DataFrame(st.session_state.model_results).T
                                best_model_name = results_df['Accuracy'].idxmax()
                                st.session_state.best_model = best_model_name
                                
                                st.success("✅ Models trained successfully using improved method!")
                                
                                # Show model performance
                                st.markdown("### Model Performance Preview")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Best Model", best_model_name)
                                with col2:
                                    st.metric("Accuracy", f"{results_df.loc[best_model_name, 'Accuracy']:.3f}")
                                with col3:
                                    st.metric("AUC-ROC", f"{results_df.loc[best_model_name, 'AUC-ROC']:.3f}")
            else:
                st.error(f"❌ {message}")
                st.info("Please upload a CSV file with the correct format and required columns.")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("Please ensure the file is a valid CSV with the required columns.")
    
    else:
        st.markdown("""
        <div class="upload-box">
        <h3>📁 No file uploaded yet</h3>
        <p>Please upload a CSV file using the file uploader above.</p>
        <p>You can download a sample template to see the required format:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create sample template (removed BP_History and Medication)
        sample_data = {
            'Age': [35, 42, 28, 55, 31],
            'Salt_Intake': [8.5, 10.2, 7.3, 9.8, 6.5],
            'Stress_Score': [6, 8, 4, 7, 5],
            'Sleep_Duration': [6.5, 5.8, 7.2, 6.0, 7.5],
            'BMI': [26.3, 29.8, 23.5, 31.2, 24.7],
            'Family_History': ['Yes', 'No', 'Yes', 'Yes', 'No'],
            'Exercise_Level': ['Moderate', 'Low', 'High', 'Low', 'Moderate'],
            'Smoking_Status': ['Non-Smoker', 'Smoker', 'Non-Smoker', 'Smoker', 'Non-Smoker'],
            'Has_Hypertension': ['No', 'Yes', 'No', 'Yes', 'No']
        }
        
        sample_df = pd.DataFrame(sample_data)
        csv = sample_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Sample Template",
            data=csv,
            file_name="hypertension_dataset_template.csv",
            mime="text/csv"
        )

# Check if data is loaded for other pages
elif st.session_state.df is None:
    st.warning("⚠️ Please upload a dataset first from the 'Upload Data' page.")
    st.stop()

# Dataset Overview Page
elif page == "📊 Dataset Overview":
    if st.session_state.df_filtered is None:
        st.warning("No data available after filtering. Adjust your filters.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(st.session_state.df):,}")
    
    with col2:
        st.metric("Filtered Samples", f"{len(st.session_state.df_filtered):,}")
    
    with col3:
        hypertension_rate = (st.session_state.df_filtered['Has_Hypertension'].str.lower() == 'yes').mean() * 100
        st.metric("Hypertension Rate", f"{hypertension_rate:.1f}%")
    
    with col4:
        avg_age = st.session_state.df_filtered['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    
    # Dataset preview
    st.markdown("### Dataset Preview")
    with st.expander("View Dataset Structure", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["First 10 Rows", "Data Types", "Missing Values", "Statistical Summary"])
        
        with tab1:
            st.dataframe(st.session_state.df_filtered.head(10), use_container_width=True)
        
        with tab2:
            # Display data types
            dtype_df = pd.DataFrame({
                'Column': st.session_state.df_filtered.columns,
                'Data Type': st.session_state.df_filtered.dtypes.values,
                'Non-Null Count': st.session_state.df_filtered.count().values,
                'Unique Values': [st.session_state.df_filtered[col].nunique() for col in st.session_state.df_filtered.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            # Check for missing values
            missing_df = pd.DataFrame({
                'Column': st.session_state.df_filtered.columns,
                'Missing Values': st.session_state.df_filtered.isnull().sum().values,
                'Missing %': (st.session_state.df_filtered.isnull().sum() / len(st.session_state.df_filtered) * 100).values
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")
        
        with tab4:
            # Statistical summary for numerical columns
            numerical_cols = st.session_state.df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                st.dataframe(st.session_state.df_filtered[numerical_cols].describe(), use_container_width=True)
            else:
                st.info("No numerical variables found.")
    
    # Distribution of key variables (changed to bar graphs)
    st.markdown("### Distribution of Key Variables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution - changed to bar graph
        age_counts = st.session_state.df_filtered['Age'].value_counts().sort_index()
        fig = px.bar(x=age_counts.index, y=age_counts.values,
                    title=f'Age Distribution ({len(st.session_state.df_filtered)} samples)',
                    labels={'x': 'Age (years)', 'y': 'Count'})
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hypertension distribution - kept as pie chart (categorical)
        hypertension_counts = st.session_state.df_filtered['Has_Hypertension'].value_counts()
        fig = px.pie(values=hypertension_counts.values, names=hypertension_counts.index,
                    title='Hypertension Distribution',
                    color_discrete_sequence=['#2ecc71', '#e74c3c'])
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)

# Exploratory Analysis Page
elif page == "🔍 Exploratory Analysis":
    if st.session_state.df_filtered is None:
        st.warning("No data available after filtering. Adjust your filters.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Display filter info
    st.markdown(f"**Analysis based on {len(st.session_state.df_filtered)} samples**")
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs([
        "📊 All Variables Analysis", 
        "👥 Demographic Factors", 
        "🏥 Health Indicators", 
        "🧬 Lifestyle Factors",
        "📈 Correlations"
    ])
    
    with analysis_tabs[0]:
        st.markdown("### Comprehensive Variable Analysis")
        
        # Select variable to analyze (removed BP_History and Medication)
        all_variables = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI',
                        'Family_History', 'Exercise_Level', 'Smoking_Status']
        
        # Only include variables that exist in the dataset
        available_vars = [var for var in all_variables if var in st.session_state.df_filtered.columns]
        
        if not available_vars:
            st.warning("No variables available for analysis.")
            st.stop()
        
        selected_var = st.selectbox("Select Variable to Analyze:", available_vars)
        
        if selected_var in ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']:
            # Numerical variable analysis - changed to bar graphs
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution by hypertension status - changed to bar graph
                # Create bins for numerical variables
                if selected_var == 'Age':
                    nbins = 20
                elif selected_var == 'BMI':
                    nbins = 20
                else:
                    nbins = 15
                
                # Create binned data for bar chart
                df_temp = st.session_state.df_filtered.copy()
                df_temp['binned'] = pd.cut(df_temp[selected_var], bins=nbins)
                # Convert intervals to string labels for JSON serialization
                df_temp['binned_label'] = df_temp['binned'].astype(str)
                
                # Group by bin label and hypertension status
                grouped = df_temp.groupby(['binned_label', 'Has_Hypertension']).size().reset_index(name='count')
                
                fig = px.bar(grouped, x='binned_label', y='count', color='Has_Hypertension',
                            title=f'{selected_var} Distribution by Hypertension Status',
                            labels={'binned_label': selected_var, 'count': 'Count', 'Has_Hypertension': 'Hypertension'},
                            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                            barmode='group')
                fig.update_layout(template=chart_theme, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot - kept as is (good for showing distribution)
                fig = px.box(st.session_state.df_filtered, x='Has_Hypertension', y=selected_var,
                            color='Has_Hypertension',
                            title=f'{selected_var} by Hypertension Status',
                            labels={'Has_Hypertension': 'Hypertension Status', selected_var: selected_var},
                            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
                fig.update_layout(template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics summary
            stats_df = st.session_state.df_filtered.groupby('Has_Hypertension')[selected_var].agg(['mean', 'std', 'min', 'max']).reset_index()
            st.dataframe(stats_df, use_container_width=True)
            
        else:
            # Categorical variable analysis - changed to bar graphs
            col1, col2 = st.columns(2)
            
            with col1:
                # Count plot - changed to bar graph
                grouped_counts = st.session_state.df_filtered.groupby([selected_var, 'Has_Hypertension']).size().reset_index(name='count')
                fig = px.bar(grouped_counts, x=selected_var, y='count', color='Has_Hypertension',
                            barmode='group',
                            title=f'{selected_var} Distribution',
                            labels={selected_var: selected_var, 'count': 'Count'},
                            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
                fig.update_layout(template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Percentage stacked bar
                cross_tab = pd.crosstab(st.session_state.df_filtered[selected_var], st.session_state.df_filtered['Has_Hypertension'], normalize='index') * 100
                cross_tab = cross_tab.reset_index()
                cross_tab_melted = cross_tab.melt(id_vars=[selected_var], value_vars=['No', 'Yes'])
                
                fig = px.bar(cross_tab_melted, x=selected_var, y='value', color='Has_Hypertension',
                            barmode='stack',
                            title=f'Hypertension Prevalence by {selected_var}',
                            labels={'value': 'Percentage (%)', selected_var: selected_var},
                            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
                fig.update_layout(template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
            
            # Contingency table
            st.markdown("##### Frequency Table")
            contingency_table = pd.crosstab(st.session_state.df_filtered[selected_var], st.session_state.df_filtered['Has_Hypertension'])
            st.dataframe(contingency_table, use_container_width=True)

# Predictive Models Page
elif page == "📈 Predictive Models":
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Predictive Model Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.markdown("""
        <div class="info-box">
        <strong>Models Not Trained Yet:</strong> 
        <p>Machine learning models need to be trained on your dataset. Please:</p>
        <ol>
        <li>Go to the "Upload Data" page</li>
        <li>Upload your dataset</li>
        <li>Click the "Train Machine Learning Models" button</li>
        </ol>
        <p>Once trained, the model performance will be displayed here.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick train button
        if st.button("🚀 Train Models Now", type="primary"):
            with st.spinner("Training models using improved method... This may take a few seconds."):
                result = train_models_improved(st.session_state.df)
                
                if result is not None:
                    st.session_state.model_results, st.session_state.feature_importances, trained_models = result
                    st.session_state.models_trained = True
                    
                    # Find best model
                    if st.session_state.model_results:
                        results_df = pd.DataFrame(st.session_state.model_results).T
                        best_model_name = results_df['Accuracy'].idxmax()
                        st.session_state.best_model = best_model_name
                        
                        st.success("✅ Models trained successfully!")
                        st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
        <strong>Model Performance Overview:</strong> This section displays the performance of various machine learning models 
        trained on your uploaded dataset to predict hypertension risk. Models are trained on data from ages 18-35.
        </div>
        """, unsafe_allow_html=True)
        
        # Display model performance
        st.markdown("### Model Performance Comparison")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(st.session_state.model_results).T.reset_index()
        results_df = results_df.rename(columns={'index': 'Model'})
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Model comparison bar chart
            fig = px.bar(results_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        barmode='group', title='Model Performance Metrics',
                        labels={'value': 'Score', 'variable': 'Metric'},
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Best model
            best_model_name = st.session_state.best_model
            best_model_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>🏆 Best Model: {best_model_name}</h4>
            <p><strong>Accuracy:</strong> {best_model_metrics['Accuracy']:.3f}</p>
            <p><strong>Precision:</strong> {best_model_metrics['Precision']:.3f}</p>
            <p><strong>Recall:</strong> {best_model_metrics['Recall']:.3f}</p>
            <p><strong>F1-Score:</strong> {best_model_metrics['F1-Score']:.3f}</p>
            <p><strong>AUC-ROC:</strong> {best_model_metrics['AUC-ROC']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Get model coefficients for visualization
        coefficients = get_model_coefficients()
        
        if coefficients:
            st.markdown("### Model Interpretability Analysis")
            
            # Create tabs for different model interpretations
            model_tabs = st.tabs(["Logistic Regression Coefficients", "Random Forest Importance", "XGBoost Importance"])
            
            with model_tabs[0]:
                if 'Logistic_Regression' in coefficients:
                    log_coefs = coefficients['Logistic_Regression']
                    
                    # Create DataFrame for visualization
                    log_coefs_df = pd.DataFrame({
                        'Feature': log_coefs.index,
                        'Coefficient': log_coefs.values
                    })
                    
                    # Add color based on coefficient sign
                    log_coefs_df['Color'] = log_coefs_df['Coefficient'].apply(lambda x: 'red' if x > 0 else 'blue')
                    log_coefs_df['Effect'] = log_coefs_df['Coefficient'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
                    
                    # Show top 15 features
                    top_features = 15
                    log_coefs_sorted = log_coefs_df.sort_values('Coefficient', ascending=False)
                    
                    # Take top positive and negative
                    top_positive = log_coefs_sorted[log_coefs_sorted['Coefficient'] > 0].head(top_features//2)
                    top_negative = log_coefs_sorted[log_coefs_sorted['Coefficient'] < 0].head(top_features//2)
                    combined = pd.concat([top_positive, top_negative]).sort_values('Coefficient')
                    
                    fig = px.bar(combined, x='Coefficient', y='Feature', orientation='h',
                                color='Effect',
                                title='Impact on Hypertension Risk (Logistic Regression)<br>Negative = Reduces Risk | Positive = Increases Risk',
                                labels={'Coefficient': 'Coefficient Value (Effect Size)', 'Feature': 'Features'},
                                color_discrete_map={'Increases Risk': '#e74c3c', 'Decreases Risk': '#3498db'},
                                hover_data=['Coefficient'])
                    fig.update_layout(template=chart_theme, height=500)
                    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show interpretation
                    st.markdown("""
                    #### Interpretation of Logistic Regression Coefficients:
                    
                    - **Positive coefficients** indicate factors that increase hypertension risk
                    - **Negative coefficients** indicate factors that decrease hypertension risk
                    - The magnitude shows the strength of the relationship
                    """)
                else:
                    st.info("Logistic Regression coefficients not available")
            
            with model_tabs[1]:
                if 'Random_Forest' in coefficients:
                    rf_importance = coefficients['Random_Forest']
                    
                    # Get top 15 features
                    top_n = 15
                    rf_top = rf_importance.head(top_n)
                    
                    fig = px.bar(x=rf_top.values, y=rf_top.index, orientation='h',
                                title=f'Top {top_n} Feature Importance - Random Forest',
                                labels={'x': 'Importance Score', 'y': 'Features'},
                                color=rf_top.values,
                                color_continuous_scale='viridis')
                    fig.update_layout(template=chart_theme, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table of top features
                    with st.expander("View Detailed Feature Importance"):
                        rf_df = pd.DataFrame({
                            'Feature': rf_importance.index,
                            'Importance': rf_importance.values
                        }).sort_values('Importance', ascending=False)
                        st.dataframe(rf_df.head(20), use_container_width=True)
                else:
                    st.info("Random Forest feature importance not available")
            
            with model_tabs[2]:
                if 'XGBoost' in coefficients:
                    xgb_importance = coefficients['XGBoost']
                    
                    # Get top 15 features
                    top_n = 15
                    xgb_top = xgb_importance.head(top_n)
                    
                    fig = px.bar(x=xgb_top.values, y=xgb_top.index, orientation='h',
                                title=f'Top {top_n} Feature Importance - XGBoost',
                                labels={'x': 'Importance Score', 'y': 'Features'},
                                color=xgb_top.values,
                                color_continuous_scale='magma')
                    fig.update_layout(template=chart_theme, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table of top features
                    with st.expander("View Detailed Feature Importance"):
                        xgb_df = pd.DataFrame({
                            'Feature': xgb_importance.index,
                            'Importance': xgb_importance.values
                        }).sort_values('Importance', ascending=False)
                        st.dataframe(xgb_df.head(20), use_container_width=True)
                else:
                    st.info("XGBoost feature importance not available")
        
        # Detailed model evaluation
        st.markdown("### Detailed Model Evaluation")
        
        if st.session_state.trained_models and st.session_state.y_test is not None:
            # Select model for detailed evaluation
            model_names = list(st.session_state.trained_models.keys())
            selected_model = st.selectbox("Select model for detailed evaluation:", model_names)
            
            if selected_model:
                model = st.session_state.trained_models[selected_model]
                
                # Get predictions
                if selected_model == "Logistic Regression":
                    X_test_data = st.session_state.X_test_scaled
                else:
                    X_test_data = st.session_state.X_test.values if hasattr(st.session_state.X_test, 'values') else st.session_state.X_test
                
                y_pred = model.predict(X_test_data)
                y_prob = model.predict_proba(X_test_data)[:, 1]
                
                # Display classification report
                report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Classification Report")
                    st.dataframe(report_df, use_container_width=True)
                
                with col2:
                    st.markdown("#### Confusion Matrix")
                    
                    cm = confusion_matrix(st.session_state.y_test, y_pred)
                    cm_df = pd.DataFrame(cm, 
                                        index=['Actual No', 'Actual Yes'],
                                        columns=['Predicted No', 'Predicted Yes'])
                    
                    fig = px.imshow(cm_df,
                                   text_auto=True,
                                   color_continuous_scale='Blues',
                                   title='Confusion Matrix',
                                   labels=dict(x="Predicted", y="Actual", color="Count"))
                    fig.update_layout(template=chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                st.markdown("#### ROC Curve")
                
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_prob)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model} (AUC = {roc_auc_score(st.session_state.y_test, y_prob):.3f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                fig.update_layout(
                    title=f'ROC Curve - {selected_model}',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template=chart_theme,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Model insights
        st.markdown("### Model Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            <div class="info-box">
            <h5>🔑 Key Insights from Your Data:</h5>
            <ul>
            <li>Models trained on ages 18-35 from your dataset</li>
            <li>Performance metrics reflect your data patterns</li>
            <li>Feature importance shows what matters most in your data</li>
            <li>Logistic Regression coefficients show direction of effects</li>
            <li>Tree-based models show relative importance of features</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div class="warning-box">
            <h5>⚠️ Important Notes:</h5>
            <ul>
            <li>Models are trained specifically on your uploaded data (ages 18-35)</li>
            <li>Performance depends on data quality and size</li>
            <li>Results are for educational/research purposes only</li>
            <li>Not for clinical decision-making</li>
            <li>Logistic Regression coefficients assume linear relationships</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# Risk Assessment Page
elif page == "🎯 Risk Assessment":
    if st.session_state.df is None:
        st.warning("Please upload a dataset first from the 'Upload Data' page.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Personalized Risk Assessment</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if not st.session_state.models_trained:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Models Not Trained</h4>
        <p>For the most accurate risk assessment, please train the machine learning models first:</p>
        <ol>
        <li>Go to the "Upload Data" page</li>
        <li>Click the "Train Machine Learning Models" button</li>
        <li>Return here for personalized risk assessment</li>
        </ol>
        <p>You can still use the calculator below, but it will use a simplified scoring system.</p>
        </div>
        """, unsafe_allow_html=True)
        use_ml_model = False
    else:
        st.markdown("""
        <div class="info-box">
        <strong>ML-Powered Risk Calculator:</strong> 
        <p>This calculator uses the trained machine learning model on your dataset for accurate risk prediction.</p>
        <p>Enter your health information below to estimate your hypertension risk.</p>
        </div>
        """, unsafe_allow_html=True)
        use_ml_model = True
    
    # Create risk assessment form (removed BP_History and Medication)
    with st.form("risk_assessment_form"):
        st.markdown("### Personal Health Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographic Information")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
            family_history = st.radio("Family History of Hypertension", ["Yes", "No"], horizontal=True)
        
        with col2:
            st.markdown("#### Health Measurements")
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
        with col3:
            st.markdown("#### Lifestyle Factors")
            exercise_level = st.select_slider("Exercise Level", options=["Low", "Moderate", "High"])
            smoking_status = st.radio("Smoking Status", ["Non-Smoker", "Smoker"], horizontal=True)
            salt_intake = st.number_input("Daily Salt Intake (grams)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
            stress_score = st.slider("Stress Level (0-10)", 0, 10, 5)
            sleep_duration = st.number_input("Sleep Duration (hours)", min_value=2.0, max_value=12.0, value=6.5, step=0.1)
        
        submitted = st.form_submit_button("Calculate My Risk Score", type="primary")
    
    if submitted:
        st.markdown("---")
        
        if use_ml_model and st.session_state.models_trained and st.session_state.trained_models:
            # Use trained ML model for prediction
            try:
                # Prepare input data (removed BP_History and Medication)
                input_data = {
                    'Age': age,
                    'Salt_Intake': salt_intake,
                    'Stress_Score': stress_score,
                    'Sleep_Duration': sleep_duration,
                    'BMI': bmi,
                    'Family_History': family_history,
                    'Exercise_Level': exercise_level,
                    'Smoking_Status': smoking_status
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Apply the same preprocessing as during training
                # Drop the same columns
                cols_to_drop = ['Age_Group', 'Age_Category', 'Patient_ID', 'BMI_Category']
                for col in cols_to_drop:
                    if col in input_df.columns:
                        input_df = input_df.drop(columns=[col])
                
                # One-hot encode like training
                input_encoded = pd.get_dummies(input_df, drop_first=False)
                
                # Align columns with training data
                training_columns = st.session_state.feature_names
                for col in training_columns:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                # Reorder columns to match training
                input_encoded = input_encoded[training_columns]
                
                # Scale the input
                input_scaled = st.session_state.scaler.transform(input_encoded)
                
                # Use the best model for prediction
                best_model_name = st.session_state.best_model
                model = st.session_state.trained_models[best_model_name]
                
                # Make prediction
                if best_model_name == "Logistic Regression":
                    prediction_proba = model.predict_proba(input_scaled)[0]
                else:
                    prediction_proba = model.predict_proba(input_encoded)[0]
                
                # Get risk score (probability of hypertension)
                risk_score = prediction_proba[1] * 100  # Probability of "Yes" class
                
                ml_based = True
                
            except Exception as e:
                st.warning(f"Could not use ML model: {str(e)}. Using simplified scoring.")
                risk_score = calculate_simplified_risk(age, bmi, family_history, 
                                                      exercise_level, smoking_status,
                                                      salt_intake, stress_score, sleep_duration)
                ml_based = False
        else:
            # Use simplified scoring (removed BP_History and Medication)
            risk_score = calculate_simplified_risk(age, bmi, family_history, 
                                                  exercise_level, smoking_status,
                                                  salt_intake, stress_score, sleep_duration)
            ml_based = False
        
        # Determine risk category
        if risk_score < 30:
            risk_category = "Low Risk"
            color = "green"
            recommendations = get_recommendations('low')
        elif risk_score < 50:
            risk_category = "Moderate Risk"
            color = "orange"
            recommendations = get_recommendations('moderate')
        elif risk_score < 70:
            risk_category = "High Risk"
            color = "red"
            recommendations = get_recommendations('high')
        else:
            risk_category = "Very High Risk"
            color = "darkred"
            recommendations = get_recommendations('very_high')
        
        # Display results (updated function signature)
        display_risk_results(risk_score, risk_category, color, recommendations, 
                           age, bmi, family_history, exercise_level, smoking_status, 
                           salt_intake, stress_score, sleep_duration, ml_based)

# About Page
elif page == "📋 About":
    st.markdown('<h2 class="sub-header">About This Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        This interactive dashboard is designed to analyze and visualize hypertension risk factors
        using your own dataset. The project provides insights into hypertension prediction 
        and personalized risk assessment based on your specific data.
        
        #### Key Features:
        
        1. **Data Upload**: Upload your own hypertension dataset in CSV format
        2. **Dataset Exploration**: Comprehensive overview with filtering capabilities
        3. **Exploratory Analysis**: Interactive visualizations of all predictors
        4. **Predictive Modeling**: Real machine learning models trained on your data (ages 18-35)
        5. **Risk Assessment**: Personalized calculator using your data patterns
        
        #### Required Data Format:
        
        Your CSV file must contain these 9 columns:
        - Age, Salt_Intake, Stress_Score, Sleep_Duration, BMI (numerical)
        - Family_History, Exercise_Level, Smoking_Status, Has_Hypertension (categorical)
        
        #### Methodology:
        
        1. **Data Validation**: Check uploaded data format and quality
        2. **Statistical Analysis**: Comprehensive EDA of all variables
        3. **Machine Learning**: Multiple models (Logistic Regression, Random Forest, XGBoost)
        - Trained on data from ages 18-35
        - One-hot encoding for categorical variables
        - Standard scaling for numerical features
        4. **Risk Calculation**: ML-based prediction using trained models
        
        #### Target Audience:
        
        - Healthcare researchers and professionals
        - Public health organizations
        - Individuals interested in hypertension analysis
        - Data science students and practitioners
        
        #### Disclaimer:
        
        This tool is for educational and informational purposes only. It is not a substitute for
        professional medical advice, diagnosis, or treatment. Always seek the advice of your
        physician or other qualified health provider with any questions you may have regarding
        a medical condition.
        """)
    
    with col2:
        st.markdown("""
        ### Technical Details
        
        #### Built With:
        
        - **Python**: Primary programming language
        - **Streamlit**: Web application framework
        - **Plotly**: Interactive visualization library
        - **Scikit-learn**: Machine learning library
        - **XGBoost**: Advanced gradient boosting library
        - **Pandas & NumPy**: Data manipulation
        
        #### Machine Learning Models:
        
        1. **Logistic Regression**
        2. **Random Forest Classifier**
        3. **XGBoost Classifier**
        
        #### Model Performance Metrics:
        
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - AUC-ROC
        
        #### Data Processing:
        
        - Real-time data validation
        - Automatic data preprocessing
        - Feature engineering
        - Missing value handling
        
        ---
        
        #### Development Features
        
        - No synthetic data - uses only your uploaded dataset
        - Real-time data filtering for all analyses
        - Interactive visualizations for all variables
        - Actual ML model training on your data
        - Personalized risk assessment
        
        #### Version Information
        
        - Current Version: 3.0.0
        - Last Updated: March 2024
        - Data Source: User-uploaded CSV files
        - Minimum Data: 50 records recommended
        
        #### Privacy Note
        
        - All data processing happens locally in your browser
        - No data is sent to external servers
        - You maintain full control of your data
        """)
    
    # Add download section
    st.markdown("---")
    st.markdown("### Resources and Downloads")
    
    # Create sample template (updated with new required columns)
    sample_data = {
        'Age': [35, 42, 28, 55, 31, 47, 39, 52, 36, 44],
        'Salt_Intake': [8.5, 10.2, 7.3, 9.8, 6.5, 11.0, 8.2, 9.5, 7.8, 10.5],
        'Stress_Score': [6, 8, 4, 7, 5, 9, 6, 8, 5, 7],
        'Sleep_Duration': [6.5, 5.8, 7.2, 6.0, 7.5, 5.5, 6.8, 6.2, 7.0, 6.5],
        'BMI': [26.3, 29.8, 23.5, 31.2, 24.7, 28.5, 26.8, 30.1, 25.4, 27.9],
        'Family_History': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
        'Exercise_Level': ['Moderate', 'Low', 'High', 'Low', 'Moderate', 
                          'Low', 'Moderate', 'Low', 'High', 'Moderate'],
        'Smoking_Status': ['Non-Smoker', 'Smoker', 'Non-Smoker', 'Smoker', 'Non-Smoker',
                          'Smoker', 'Non-Smoker', 'Smoker', 'Non-Smoker', 'Smoker'],
        'Has_Hypertension': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
    }
    
    sample_df = pd.DataFrame(sample_data)
    csv = sample_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Complete Sample Dataset (10 records)",
        data=csv,
        file_name="hypertension_sample_dataset.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**Hypertension Risk Predictor Dashboard**")
    st.markdown("Version 3.0 | March 2024")

with footer_col2:
    st.markdown("**Upload Your Own Data**")
    st.markdown("No synthetic data used")

with footer_col3:
    st.markdown("**Built with Streamlit**")
    st.markdown("Data Science Project")