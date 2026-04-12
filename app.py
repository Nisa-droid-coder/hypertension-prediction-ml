import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Import XGBoost or use fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    st.warning("XGBoost not installed. Install it using: pip install xgboost")
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
if 'X_train_scaled' not in st.session_state:
    st.session_state.X_train_scaled = None
if 'X_test_scaled' not in st.session_state:
    st.session_state.X_test_scaled = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_encoded' not in st.session_state:
    st.session_state.X_encoded = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'cv_scores' not in st.session_state:
    st.session_state.cv_scores = None
if 'importance_df' not in st.session_state:
    st.session_state.importance_df = None

# Global chart theme variable
chart_theme = "plotly_white"

def create_age_bins(age_min, age_max, bin_size=5):
    """Create age bins with specified bin size"""
    bins = list(range(age_min, age_max + bin_size, bin_size))
    if bins[-1] < age_max:
        bins.append(age_max + 1)
    
    labels = []
    for i in range(len(bins)-1):
        if i == len(bins)-2:
            labels.append(f"{bins[i]}-{bins[i+1]}")
        else:
            labels.append(f"{bins[i]}-{bins[i+1]-1}")
    
    return bins, labels

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
    df_processed['Age_Group'] = df_processed['Age_Group'].astype(str)
    
    # Create BMI categories
    df_processed['BMI_Category'] = pd.cut(df_processed['BMI'], 
                                        bins=[0, 18.5, 25, 30, 100],
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df_processed['BMI_Category'] = df_processed['BMI_Category'].astype(str)
    
    # Clean categorical variables
    categorical_cols = ['Family_History', 'Exercise_Level', 'Smoking_Status', 'Has_Hypertension']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).str.strip()
    
    return df_processed

def train_models_improved(df):
    """Train machine learning models with proper cross-validation and NO data leakage"""
    try:
        # =================================================================
        # 1. DATA PREPARATION & LEAKAGE PREVENTION
        # =================================================================
        # Filter data for young adults (18-35) as per FYP objectives
        df_young = df[(df['Age'] >= 18) & (df['Age'] <= 35)].copy()
        
        if len(df_young) < 50:
            st.warning(f"Only {len(df_young)} samples in age range 18-35. Using full dataset instead.")
            df_young = df.copy()
        
        # LEAKAGE PREVENTION:
        # We drop 'BP_History' and 'Medication' if they exist.
        # Reasoning: Including 'BP_History' (past diagnosis) to predict 'Has_Hypertension' 
        # is circular logic and causes the "overfitting" 99% accuracy.
        # Removing them forces the model to learn from LIFESTYLE (Salt, Stress, Sleep, etc.)
        
        columns_to_drop = ['Has_Hypertension']
        if 'BP_History' in df_young.columns:
            columns_to_drop.append('BP_History')
            st.info("✅ Detected and removed 'BP_History' column to prevent data leakage")
        if 'Medication' in df_young.columns:
            columns_to_drop.append('Medication')
            st.info("✅ Detected and removed 'Medication' column to prevent data leakage")
        
        # Define lifestyle features (all numerical + categorical lifestyle factors)
        lifestyle_features = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI', 
                             'Family_History', 'Exercise_Level', 'Smoking_Status']
        
        # Check which features are available
        available_features = [col for col in lifestyle_features if col in df_young.columns]
        
        if len(available_features) != len(lifestyle_features):
            missing = set(lifestyle_features) - set(available_features)
            st.warning(f"Missing features: {missing}. Using available features only.")
        
        # Prepare X and y
        X = df_young[available_features].copy()
        y = df_young['Has_Hypertension'].map(lambda x: 1 if str(x).lower() == 'yes' else 0)
        
        # Check if y has both classes
        if y.nunique() < 2:
            st.error(f"Target variable has only {y.nunique()} class(es). Need both 0 and 1.")
            return None
        
        st.info(f"Training data shape: {X.shape}, Classes: 0={sum(y==0)}, 1={sum(y==1)}")
        
        # Encode categorical lifestyle variables
        label_encoders = {}
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        feature_names = X.columns.tolist()
        
        # Scale data for Logistic Regression (ensures fair comparison)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Split for final evaluation (80:20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train_scaled, X_test_scaled, _, _ = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # =================================================================
        # 2. MODEL DEFINITIONS & TRAINING
        # =================================================================
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                n_estimators=100
            )
        
        # =================================================================
        # 3. FAIR COMPARISON: TRAIN, CROSS-VALIDATE, AND EVALUATE
        # =================================================================
        trained_models = {}
        results = {}
        cv_scores_dict = {}
        feature_importances = {}
        
        for name, model in models.items():
            st.write(f"Training {name}...")
            
            # Use scaled data for LogReg, raw data for Tree-based models
            curr_X_train = X_train_scaled if name == "Logistic Regression" else X_train
            curr_X_test = X_test_scaled if name == "Logistic Regression" else X_test
            
            # Stratified 5-Fold Cross-Validation (Rigor Check)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, curr_X_train, y_train, cv=cv, scoring='accuracy')
            cv_scores_dict[name] = cv_scores.mean()
            
            # Train the model
            model.fit(curr_X_train, y_train)
            trained_models[name] = model
            
            # Predict on test set
            y_pred = model.predict(curr_X_test)
            y_prob = model.predict_proba(curr_X_test)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else 0.5
            
            results[name] = {
                'Accuracy': acc,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': auc,
                'CV_Accuracy': cv_scores.mean(),
                'CV_Std': cv_scores.std()
            }
            
            # Get feature importance
            if name == "Logistic Regression":
                feature_importances[name] = dict(zip(feature_names, np.abs(model.coef_[0])))
            elif hasattr(model, 'feature_importances_'):
                feature_importances[name] = dict(zip(feature_names, model.feature_importances_))
        
        # =================================================================
        # 4. CLINICAL ACTIONABILITY TABLE
        # =================================================================
        importance_data = {'Feature': feature_names}
        
        for model_name in models.keys():
            if model_name in feature_importances:
                short_name = model_name.split()[0]
                importance_data[f'{short_name}_Importance'] = [
                    feature_importances[model_name].get(f, 0) for f in feature_names
                ]
        
        # Calculate Population Risk Thresholds (mean values for hypertensive group)
        thresholds = []
        for feature in feature_names:
            if feature in ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']:
                hypertensive_vals = df_young[df_young['Has_Hypertension'].str.lower() == 'yes'][feature]
                h_mean = hypertensive_vals.mean() if len(hypertensive_vals) > 0 else 0
                thresholds.append(round(h_mean, 2))
            else:
                thresholds.append("N/A (Categorical)")
        
        importance_data['Risk_Threshold_Avg'] = thresholds
        importance_df = pd.DataFrame(importance_data)
        
        # Sort by Random Forest importance if available
        if 'RF_Importance' in importance_data:
            importance_df = importance_df.sort_values(by='RF_Importance', ascending=False)
        elif 'Log_Importance' in importance_data:
            importance_df = importance_df.sort_values(by='Log_Importance', ascending=False)
        
        # Find best model based on CV accuracy (more reliable than test accuracy)
        best_model_name = max(cv_scores_dict, key=cv_scores_dict.get)
        
        # Store everything in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.scaler = scaler
        st.session_state.trained_models = trained_models
        st.session_state.feature_names = feature_names
        st.session_state.label_encoders = label_encoders
        st.session_state.cv_scores = cv_scores_dict
        st.session_state.importance_df = importance_df
        st.session_state.best_model = best_model_name
        st.session_state.model_results = results
        
        # Display CV results for transparency
        st.info(f"📊 Cross-Validation Results (5-fold stratified on training data):")
        for name in cv_scores_dict:
            auc_value = results[name]['AUC-ROC']
            auc_indicator = "✓" if auc_value <= 0.90 else "⚠️" if auc_value <= 0.95 else "❌"
            st.write(f"  • {name}: CV={cv_scores_dict[name]:.3f}, Test Acc={results[name]['Accuracy']:.3f}, AUC={auc_value:.3f} {auc_indicator}")
        
        # Warning for high AUC
        high_auc_models = [name for name, metrics in results.items() if metrics['AUC-ROC'] > 0.95]
        if high_auc_models:
            st.warning(f"""
            ⚠️ **High AUC-ROC Detected!** 
            
            Models with AUC > 0.95: {', '.join(high_auc_models)}
            
            This might indicate:
            - Very strong predictive features
            - Potential overfitting (if dataset is small)
            - Perfect separation in the data
            
            The dashboard will use {best_model_name} for risk assessment.
            """)
        
        return results, feature_importances, trained_models
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def safe_predict(input_data, model_name, model, scaler, label_encoders, feature_names):
    """Safely predict with proper encoding and error handling"""
    try:
        # Create a copy of input data
        input_df = input_data.copy()
        
        # Encode categorical variables safely
        for col in ['Family_History', 'Exercise_Level', 'Smoking_Status']:
            if col in input_df.columns and col in label_encoders:
                try:
                    # Transform using fitted encoder
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except ValueError as e:
                    # Handle unseen categories
                    st.warning(f"Unseen category in {col}. Using default encoding.")
                    # Create a mapping from fitted classes
                    fitted_classes = label_encoders[col].classes_.tolist()
                    # Map unknown values to the most common class
                    default_value = 0
                    input_df[col] = input_df[col].apply(
                        lambda x: label_encoders[col].transform([x])[0] if str(x) in fitted_classes else default_value
                    )
        
        # Ensure all features are present
        missing_features = set(feature_names) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
            st.warning(f"Missing feature '{feature}' added with default value 0")
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Scale if using Logistic Regression
        if model_name == "Logistic Regression" and scaler is not None:
            input_scaled = scaler.transform(input_df)
            prediction_proba = model.predict_proba(input_scaled)[0]
        else:
            prediction_proba = model.predict_proba(input_df)[0]
        
        return prediction_proba[1] * 100
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def plot_learning_curve(model, X, y, model_name, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """Plot learning curve starting from small sample sizes"""
    try:
        # Generate learning curve data with more points starting from small samples
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, 
            cv=cv, 
            n_jobs=-1,
            train_sizes=train_sizes,
            scoring='accuracy',
            shuffle=True,
            random_state=42
        )
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create interactive plotly figure
        fig = go.Figure()
        
        # Add training accuracy line
        fig.add_trace(go.Scatter(
            x=train_sizes_abs, 
            y=train_mean,
            mode='lines+markers',
            name='Training Accuracy',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Training Samples: %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
        ))
        
        # Add validation accuracy line
        fig.add_trace(go.Scatter(
            x=train_sizes_abs, 
            y=test_mean,
            mode='lines+markers',
            name='Validation Accuracy',
            line=dict(color='orange', width=2),
            marker=dict(size=8),
            hovertemplate='Training Samples: %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
        ))
        
        # Add confidence bands for training
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Training ±1 std',
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Add confidence bands for validation
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Validation ±1 std',
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Add a horizontal line at 0.5 (random guessing)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Random Guessing (0.5)", 
                     annotation_position="bottom right")
        
        # Update layout
        fig.update_layout(
            title=f'Learning Curve: {model_name}',
            xaxis_title='Number of Training Samples',
            yaxis_title='Accuracy Score',
            template=chart_theme,
            height=450,
            hovermode='closest',
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, 1.05]
            )
        )
        
        # Add annotation for best performance
        best_val_acc = test_mean.max()
        best_val_samples = train_sizes_abs[test_mean.argmax()]
        fig.add_annotation(
            x=best_val_samples,
            y=best_val_acc,
            text=f"Best: {best_val_acc:.3f} at {best_val_samples:.0f} samples",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="green",
            ax=20,
            ay=-30,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='green',
            borderwidth=1
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate learning curve for {model_name}: {str(e)}")
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
                np.abs(log_model.coef_[0]), 
                index=st.session_state.feature_names
            ).sort_values(ascending=False)
    
    # Random Forest Feature Importance
    if "Random Forest" in st.session_state.trained_models:
        rf_model = st.session_state.trained_models["Random Forest"]
        if hasattr(rf_model, 'feature_importances_'):
            coefficients['Random_Forest'] = pd.Series(
                rf_model.feature_importances_,
                index=st.session_state.feature_names
            ).sort_values(ascending=False)
    
    # XGBoost Feature Importance
    if XGB_AVAILABLE and "XGBoost" in st.session_state.trained_models:
        xgb_model = st.session_state.trained_models["XGBoost"]
        if hasattr(xgb_model, 'feature_importances_'):
            coefficients['XGBoost'] = pd.Series(
                xgb_model.feature_importances_,
                index=st.session_state.feature_names
            ).sort_values(ascending=False)
    
    return coefficients

def calculate_simplified_risk(age, bmi, family_history, exercise_level, smoking_status, salt_intake, stress_score, sleep_duration):
    """Calculate risk using simplified scoring system"""
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
                        stress_score, sleep_duration, ml_based=False, model_name=None):
    """Display risk assessment results"""
    st.markdown("## 📊 Your Risk Assessment Results")
    
    if ml_based and model_name:
        st.info(f"✅ Using {model_name} ML model trained on your dataset")
    elif ml_based:
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
            for factor in risk_factors[:5]:
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
        
        dataset_stats = st.session_state.df['Has_Hypertension'].str.lower() == 'yes'
        dataset_hypertension_rate = dataset_stats.mean() * 100
        
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
    {'Using ML Model: ' + model_name if ml_based and model_name else 'Using Simplified Scoring'}
    
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
        
        # Age bin size selector
        bin_size = st.slider(
            "Age Bin Size (years):",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Select the size of age groups for bar charts"
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
    <li>Must contain these columns (exact names):</li>
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
    <p><strong>Note:</strong> BP_History and Medication columns are automatically detected and excluded from model training to prevent data leakage.</p>
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
                if st.button("🚀 Train Machine Learning Models", type="primary"):
                    with st.spinner("Training models with data leakage prevention... This may take a few seconds."):
                        result = train_models_improved(df_processed)
                        
                        if result is not None:
                            st.session_state.model_results, st.session_state.feature_importances, trained_models = result
                            st.session_state.models_trained = True
                            
                            st.success("✅ Models trained successfully!")
                            st.success("Models are trained on LIFESTYLE factors only (BP_History and Medication automatically excluded to prevent data leakage)")
                            
                            # Show model performance
                            if st.session_state.model_results:
                                results_df = pd.DataFrame(st.session_state.model_results).T
                                st.markdown("### Model Performance Preview")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Best Model (by CV)", st.session_state.best_model)
                                with col2:
                                    st.metric("Best CV Accuracy", f"{results_df.loc[st.session_state.best_model, 'CV_Accuracy']:.3f}")
                                with col3:
                                    st.metric("Best Test Accuracy", f"{results_df.loc[st.session_state.best_model, 'Accuracy']:.3f}")
                        else:
                            st.error("Model training failed. Please check your data.")
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
        
        # Create sample template
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
            dtype_df = pd.DataFrame({
                'Column': st.session_state.df_filtered.columns,
                'Data Type': st.session_state.df_filtered.dtypes.values,
                'Non-Null Count': st.session_state.df_filtered.count().values,
                'Unique Values': [st.session_state.df_filtered[col].nunique() for col in st.session_state.df_filtered.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
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
            numerical_cols = st.session_state.df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                st.dataframe(st.session_state.df_filtered[numerical_cols].describe(), use_container_width=True)
            else:
                st.info("No numerical variables found.")

# Exploratory Analysis Page
elif page == "🔍 Exploratory Analysis":
    if st.session_state.df_filtered is None:
        st.warning("No data available after filtering. Adjust your filters.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown(f"**Analysis based on {len(st.session_state.df_filtered)} samples**")
    st.markdown(f"**Age bins: {bin_size}-year intervals**")
    
    # Create age bins based on filtered data
    age_min_filtered = int(st.session_state.df_filtered['Age'].min())
    age_max_filtered = int(st.session_state.df_filtered['Age'].max())
    bins, age_labels = create_age_bins(age_min_filtered, age_max_filtered, bin_size)
    
    # Add age category column
    df_for_chart = st.session_state.df_filtered.copy()
    df_for_chart['Age_Category'] = pd.cut(df_for_chart['Age'], bins=bins, labels=age_labels, right=False)
    df_for_chart['Age_Category'] = df_for_chart['Age_Category'].astype(str)
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs([
        "📊 Salt Intake Analysis", 
        "👥 BMI Analysis", 
        "🏥 Stress Score Analysis", 
        "🧬 Sleep Duration Analysis",
        "📈 Hypertension by Age",
        "🏷️ Categorical Analysis"
    ])
    
    with analysis_tabs[0]:
        st.markdown("### Salt Intake Distribution by Age Group")
        
        def categorize_salt_intake(salt):
            if salt < 4:
                return 'Low (<4 g/day)'
            elif salt <= 6:
                return 'Moderate (4-6 g/day)'
            elif salt <= 8:
                return 'High (6-8 g/day)'
            else:
                return 'Very High (>8 g/day)'
        
        df_for_chart['Salt_Category'] = df_for_chart['Salt_Intake'].apply(categorize_salt_intake)
        
        salt_hypertension = df_for_chart.groupby(['Age_Category', 'Has_Hypertension', 'Salt_Category']).size().reset_index(name='count')
        
        fig = px.bar(salt_hypertension, x='Age_Category', y='count', color='Salt_Category',
                    facet_col='Has_Hypertension',
                    title='Salt Intake Distribution by Age Group and Hypertension Status',
                    labels={'Age_Category': 'Age Category', 'count': 'Number of Patients', 'Salt_Category': 'Salt Intake Level'},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    barmode='group')
        fig.update_layout(template=chart_theme, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Salt Intake Statistics by Age Group")
        salt_stats = df_for_chart.groupby('Age_Category')['Salt_Intake'].agg(['mean', 'std', 'min', 'max']).reset_index()
        st.dataframe(salt_stats, use_container_width=True)
    
    with analysis_tabs[1]:
        st.markdown("### BMI Distribution by Age Group")
        
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        
        df_for_chart['BMI_Category_Display'] = df_for_chart['BMI'].apply(categorize_bmi)
        
        bmi_hypertension = df_for_chart.groupby(['Age_Category', 'Has_Hypertension', 'BMI_Category_Display']).size().reset_index(name='count')
        
        fig = px.bar(bmi_hypertension, x='Age_Category', y='count', color='BMI_Category_Display',
                    facet_col='Has_Hypertension',
                    title='BMI Distribution by Age Group and Hypertension Status',
                    labels={'Age_Category': 'Age Category', 'count': 'Number of Patients', 'BMI_Category_Display': 'BMI Category'},
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    barmode='group')
        fig.update_layout(template=chart_theme, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### BMI Statistics by Age Group")
        bmi_stats = df_for_chart.groupby('Age_Category')['BMI'].agg(['mean', 'std', 'min', 'max']).reset_index()
        st.dataframe(bmi_stats, use_container_width=True)
    
    with analysis_tabs[2]:
        st.markdown("### Stress Score Distribution by Age Group")
        
        def categorize_stress(score):
            if score <= 3:
                return 'Low (0-3)'
            elif score <= 6:
                return 'Moderate (4-6)'
            elif score <= 8:
                return 'High (7-8)'
            else:
                return 'Very High (9-10)'
        
        df_for_chart['Stress_Category'] = df_for_chart['Stress_Score'].apply(categorize_stress)
        
        stress_hypertension = df_for_chart.groupby(['Age_Category', 'Has_Hypertension', 'Stress_Category']).size().reset_index(name='count')
        
        fig = px.bar(stress_hypertension, x='Age_Category', y='count', color='Stress_Category',
                    facet_col='Has_Hypertension',
                    title='Stress Score Distribution by Age Group and Hypertension Status',
                    labels={'Age_Category': 'Age Category', 'count': 'Number of Patients', 'Stress_Category': 'Stress Level'},
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    barmode='group')
        fig.update_layout(template=chart_theme, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Stress Score Statistics by Age Group")
        stress_stats = df_for_chart.groupby('Age_Category')['Stress_Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
        st.dataframe(stress_stats, use_container_width=True)
    
    with analysis_tabs[3]:
        st.markdown("### Sleep Duration Distribution by Age Group")
        
        def categorize_sleep(sleep):
            if sleep < 6:
                return 'Insufficient (<6h)'
            elif sleep <= 8:
                return 'Optimal (6-8h)'
            else:
                return 'Excessive (>8h)'
        
        df_for_chart['Sleep_Category'] = df_for_chart['Sleep_Duration'].apply(categorize_sleep)
        
        sleep_hypertension = df_for_chart.groupby(['Age_Category', 'Has_Hypertension', 'Sleep_Category']).size().reset_index(name='count')
        
        fig = px.bar(sleep_hypertension, x='Age_Category', y='count', color='Sleep_Category',
                    facet_col='Has_Hypertension',
                    title='Sleep Duration Distribution by Age Group and Hypertension Status',
                    labels={'Age_Category': 'Age Category', 'count': 'Number of Patients', 'Sleep_Category': 'Sleep Duration'},
                    color_discrete_sequence=px.colors.qualitative.Dark2,
                    barmode='group')
        fig.update_layout(template=chart_theme, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Sleep Duration Statistics by Age Group")
        sleep_stats = df_for_chart.groupby('Age_Category')['Sleep_Duration'].agg(['mean', 'std', 'min', 'max']).reset_index()
        st.dataframe(sleep_stats, use_container_width=True)
    
    with analysis_tabs[4]:
        st.markdown("### Hypertension Prevalence by Age Group")
        
        hypertension_by_age = df_for_chart.groupby('Age_Category')['Has_Hypertension'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        hypertension_by_age.columns = ['Age_Category', 'Hypertension_Rate']
        
        count_by_age = df_for_chart.groupby('Age_Category').size().reset_index(name='Total_Patients')
        hypertension_by_age = hypertension_by_age.merge(count_by_age, on='Age_Category')
        
        fig = px.bar(hypertension_by_age, x='Age_Category', y='Hypertension_Rate',
                    title='Hypertension Rate by Age Group',
                    labels={'Age_Category': 'Age Category', 'Hypertension_Rate': 'Hypertension Rate (%)'},
                    text='Hypertension_Rate',
                    color='Hypertension_Rate',
                    color_continuous_scale='RdYlGn_r')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(template=chart_theme, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Detailed Breakdown by Age Group")
        st.dataframe(hypertension_by_age, use_container_width=True)
    
    with analysis_tabs[5]:
        st.markdown("### Categorical Variables Analysis")
        st.markdown("Analysis of Family History, Exercise Level, and Smoking Status")
        
        categorical_vars = ['Family_History', 'Exercise_Level', 'Smoking_Status']
        
        for cat_var in categorical_vars:
            if cat_var in df_for_chart.columns:
                st.markdown(f"#### {cat_var.replace('_', ' ')} Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution of categorical variable
                    cat_counts = df_for_chart[cat_var].value_counts().reset_index()
                    cat_counts.columns = [cat_var, 'Count']
                    
                    fig = px.bar(cat_counts, x=cat_var, y='Count',
                                title=f'Distribution of {cat_var.replace("_", " ")}',
                                labels={cat_var: cat_var.replace("_", " "), 'Count': 'Number of Patients'},
                                color=cat_var,
                                color_discrete_sequence=px.colors.qualitative.Set1)
                    fig.update_layout(template=chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Hypertension rate by categorical variable
                    hypertension_by_cat = df_for_chart.groupby(cat_var)['Has_Hypertension'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
                    hypertension_by_cat.columns = [cat_var, 'Hypertension_Rate']
                    
                    fig = px.bar(hypertension_by_cat, x=cat_var, y='Hypertension_Rate',
                                title=f'Hypertension Rate by {cat_var.replace("_", " ")}',
                                labels={cat_var: cat_var.replace("_", " "), 'Hypertension_Rate': 'Hypertension Rate (%)'},
                                text='Hypertension_Rate',
                                color=cat_var,
                                color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(template=chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cross tabulation
                st.markdown(f"##### Cross Tabulation: {cat_var} vs Hypertension")
                cross_tab = pd.crosstab(df_for_chart[cat_var], df_for_chart['Has_Hypertension'], normalize='index') * 100
                st.dataframe(cross_tab, use_container_width=True)
                
                st.markdown("---")

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
        <p><strong>Note:</strong> Models are trained on LIFESTYLE factors only. BP_History and Medication are automatically excluded to prevent data leakage.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train Models Now", type="primary"):
            with st.spinner("Training models with data leakage prevention... This may take a few seconds."):
                result = train_models_improved(st.session_state.df)
                
                if result is not None:
                    st.session_state.model_results, st.session_state.feature_importances, trained_models = result
                    st.session_state.models_trained = True
                    
                    st.success("✅ Models trained successfully!")
                    st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
        <strong>Model Performance Overview:</strong> Models are trained on LIFESTYLE factors only (Salt Intake, Stress, Sleep, BMI, Exercise, Smoking, Family History).
        <br><strong>Data Leakage Prevention:</strong> BP_History and Medication are automatically excluded to ensure models learn from actual lifestyle predictors.
        <br><strong>Best Model Selection:</strong> The model with highest cross-validation accuracy is used for risk assessment.
        </div>
        """, unsafe_allow_html=True)
        
        # Display model performance
        st.markdown("### Model Performance Comparison")
        
        results_df = pd.DataFrame(st.session_state.model_results).T.reset_index()
        results_df = results_df.rename(columns={'index': 'Model'})
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(results_df, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Accuracy'],
                        barmode='group', title='Model Performance Metrics',
                        labels={'value': 'Score', 'variable': 'Metric'},
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            best_model_name = st.session_state.best_model
            best_model_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>🏆 Best Model: {best_model_name}</h4>
            <p><strong>CV Accuracy (5-fold):</strong> {best_model_metrics['CV_Accuracy']:.3f}</p>
            <p><strong>Test Accuracy:</strong> {best_model_metrics['Accuracy']:.3f}</p>
            <p><strong>Precision:</strong> {best_model_metrics['Precision']:.3f}</p>
            <p><strong>Recall:</strong> {best_model_metrics['Recall']:.3f}</p>
            <p><strong>F1-Score:</strong> {best_model_metrics['F1-Score']:.3f}</p>
            <p><strong>AUC-ROC:</strong> {best_model_metrics['AUC-ROC']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clinical Actionability Table
        if st.session_state.importance_df is not None:
            st.markdown("### Clinical Actionability Table")
            st.markdown("Population Risk Thresholds (Mean values for Hypertensive group)")
            st.dataframe(st.session_state.importance_df, use_container_width=True)
        
        # Learning Curves - Display all models (starting from small sample sizes)
        st.markdown("### Learning Curves - All Models")
        st.markdown("Shows model performance as training data size increases (starting from small sample sizes)")
        
        if st.session_state.trained_models and st.session_state.X_train is not None and st.session_state.y_train is not None:
            models_to_plot = list(st.session_state.trained_models.keys())
            
            for name in models_to_plot:
                st.markdown(f"#### {name}")
                model = st.session_state.trained_models[name]
                
                # Select the correct training data
                if name == "Logistic Regression":
                    X_curr = st.session_state.X_train_scaled
                else:
                    X_curr = st.session_state.X_train
                
                if X_curr is not None and len(X_curr) > 0:
                    # Create train sizes starting from very small (10% of data or minimum 10 samples)
                    n_samples = len(X_curr)
                    # Create 15 points from very small to full dataset
                    train_sizes = np.linspace(0.05, 1.0, 15)  # Start from 5% of data
                    
                    # Ensure minimum 5 samples for early learning curve points
                    fig = plot_learning_curve(model, X_curr, st.session_state.y_train, name, 
                                             cv=min(5, len(np.unique(st.session_state.y_train))),
                                             train_sizes=train_sizes)
                    
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        st.caption(f"""
                        **Learning Curve Interpretation for {name}:**
                        - X-axis: Number of training samples (from {int(n_samples * 0.05)} to {n_samples})
                        - Y-axis: Accuracy score (0 to 1, with 0.5 being random guessing)
                        - Blue line: Training accuracy (how well model fits training data)
                        - Orange line: Validation accuracy (how well model generalizes to unseen data)
                        - Shaded areas: ±1 standard deviation across cross-validation folds
                        - **Gap between lines indicates overfitting** (training accuracy much higher than validation)
                        - **Lines converging at high values (>0.8) indicates good generalization**
                        """)
                    else:
                        st.warning(f"Could not generate learning curve for {name}")
                else:
                    st.warning(f"Insufficient data for {name} learning curve")
                st.markdown("---")
        else:
            st.warning("Model training data not available. Please retrain models.")
        
        # Confusion Matrices - Display all models
        st.markdown("### Confusion Matrices - All Models")
        st.markdown("Shows prediction performance for each model")
        
        if st.session_state.trained_models and st.session_state.y_test is not None:
            models_to_show = list(st.session_state.trained_models.keys())
            
            for name in models_to_show:
                st.markdown(f"#### {name}")
                model = st.session_state.trained_models[name]
                
                # Select the correct test data
                if name == "Logistic Regression":
                    X_test_curr = st.session_state.X_test_scaled
                else:
                    X_test_curr = st.session_state.X_test
                
                if X_test_curr is not None and len(X_test_curr) > 0:
                    try:
                        y_pred = model.predict(X_test_curr)
                        
                        cm = confusion_matrix(st.session_state.y_test, y_pred)
                        cm_df = pd.DataFrame(cm, 
                                            index=['Actual No', 'Actual Yes'],
                                            columns=['Predicted No', 'Predicted Yes'])
                        
                        fig = px.imshow(cm_df,
                                       text_auto=True,
                                       color_continuous_scale='Blues',
                                       title=f'{name} - Confusion Matrix',
                                       labels=dict(x="Predicted", y="Actual", color="Count"))
                        fig.update_layout(template=chart_theme, height=450)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate confusion matrix for {name}: {str(e)}")
                else:
                    st.warning(f"Test data not available for {name}")
                st.markdown("---")
        else:
            st.warning("Test data not available. Please retrain models.")
        
        # Feature Importance
        coefficients = get_model_coefficients()
        
        if coefficients:
            st.markdown("### Feature Importance Analysis")
            
            model_tabs = st.tabs(["Logistic Regression", "Random Forest", "XGBoost"])
            
            with model_tabs[0]:
                if 'Logistic_Regression' in coefficients:
                    log_importance = coefficients['Logistic_Regression'].head(10)
                    
                    fig = px.bar(x=log_importance.values, y=log_importance.index, orientation='h',
                                title='Top 10 Feature Importance - Logistic Regression (Absolute Coefficients)',
                                labels={'x': 'Absolute Coefficient Value', 'y': 'Features'},
                                color=log_importance.values,
                                color_continuous_scale='Blues')
                    fig.update_layout(template=chart_theme, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Logistic Regression coefficients not available")
            
            with model_tabs[1]:
                if 'Random_Forest' in coefficients:
                    rf_importance = coefficients['Random_Forest'].head(10)
                    
                    fig = px.bar(x=rf_importance.values, y=rf_importance.index, orientation='h',
                                title='Top 10 Feature Importance - Random Forest',
                                labels={'x': 'Importance Score', 'y': 'Features'},
                                color=rf_importance.values,
                                color_continuous_scale='Greens')
                    fig.update_layout(template=chart_theme, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Random Forest feature importance not available")
            
            with model_tabs[2]:
                if XGB_AVAILABLE and 'XGBoost' in coefficients:
                    xgb_importance = coefficients['XGBoost'].head(10)
                    
                    fig = px.bar(x=xgb_importance.values, y=xgb_importance.index, orientation='h',
                                title='Top 10 Feature Importance - XGBoost',
                                labels={'x': 'Importance Score', 'y': 'Features'},
                                color=xgb_importance.values,
                                color_continuous_scale='Oranges')
                    fig.update_layout(template=chart_theme, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                elif not XGB_AVAILABLE:
                    st.info("XGBoost not installed. Install it using: pip install xgboost")
                else:
                    st.info("XGBoost feature importance not available")
        
        # Model insights
        st.markdown("### Model Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            <div class="info-box">
            <h5>🔑 Key Insights from Your Data:</h5>
            <ul>
            <li>Models trained on ages 18-35 from your dataset</li>
            <li><strong>No data leakage:</strong> BP_History and Medication excluded</li>
            <li>Models learn from LIFESTYLE factors only</li>
            <li>5-Fold Stratified Cross-Validation ensures robustness</li>
            <li>Learning curves start from small sample sizes (5% of data)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div class="warning-box">
            <h5>⚠️ Important Notes:</h5>
            <ul>
            <li>Models are trained specifically on your uploaded data</li>
            <li>Performance depends on data quality and size</li>
            <li>Results are for educational/research purposes only</li>
            <li>Not for clinical decision-making</li>
            <li>Risk thresholds show population averages</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# Risk Assessment Page
elif page == "🎯 Risk Assessment":
    if st.session_state.df is None:
        st.warning("Please upload a dataset first from the 'Upload Data' page.")
        st.stop()
    
    st.markdown('<h2 class="sub-header">Personalized Risk Assessment</h2>', unsafe_allow_html=True)
    
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
        st.markdown(f"""
        <div class="info-box">
        <strong>ML-Powered Risk Calculator:</strong> 
        <p>This calculator uses the trained <strong>{st.session_state.best_model}</strong> model on your dataset for accurate risk prediction.</p>
        <p>Models are trained on LIFESTYLE factors only (BP_History and Medication automatically excluded to prevent data leakage).</p>
        <p><strong>Data Split:</strong> 80% training, 20% testing | <strong>Cross-Validation:</strong> 5-fold Stratified</p>
        </div>
        """, unsafe_allow_html=True)
        use_ml_model = True
    
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
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'Age': age,
            'Salt_Intake': salt_intake,
            'Stress_Score': stress_score,
            'Sleep_Duration': sleep_duration,
            'BMI': bmi,
            'Family_History': family_history,
            'Exercise_Level': exercise_level,
            'Smoking_Status': smoking_status
        }])
        
        if use_ml_model and st.session_state.models_trained and st.session_state.trained_models:
            try:
                best_model_name = st.session_state.best_model
                model = st.session_state.trained_models[best_model_name]
                
                # Use safe prediction function
                risk_score = safe_predict(
                    input_data, 
                    best_model_name, 
                    model, 
                    st.session_state.scaler, 
                    st.session_state.label_encoders, 
                    st.session_state.feature_names
                )
                
                if risk_score is not None:
                    ml_based = True
                    model_used = best_model_name
                    st.success(f"✅ Successfully used {best_model_name} model for prediction!")
                else:
                    # Fallback to simplified scoring
                    risk_score = calculate_simplified_risk(age, bmi, family_history, 
                                                          exercise_level, smoking_status,
                                                          salt_intake, stress_score, sleep_duration)
                    ml_based = False
                    model_used = None
                    st.warning("Falling back to simplified scoring due to prediction error.")
                    
            except Exception as e:
                st.warning(f"Could not use ML model: {str(e)}. Using simplified scoring.")
                risk_score = calculate_simplified_risk(age, bmi, family_history, 
                                                      exercise_level, smoking_status,
                                                      salt_intake, stress_score, sleep_duration)
                ml_based = False
                model_used = None
        else:
            risk_score = calculate_simplified_risk(age, bmi, family_history, 
                                                  exercise_level, smoking_status,
                                                  salt_intake, stress_score, sleep_duration)
            ml_based = False
            model_used = None
        
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
        
        display_risk_results(risk_score, risk_category, color, recommendations, 
                           age, bmi, family_history, exercise_level, smoking_status, 
                           salt_intake, stress_score, sleep_duration, ml_based, model_used)

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
        3. **Exploratory Analysis**: Interactive visualizations of all predictors with age-based binning and categorical analysis
        4. **Predictive Modeling**: Real machine learning models trained on your data (ages 18-35)
        5. **Risk Assessment**: Personalized calculator using your data patterns with best model selection
        
        #### Data Leakage Prevention:
        
        - **BP_History** and **Medication** are automatically detected and excluded from model training
        - This prevents circular logic and ensures models learn from actual LIFESTYLE predictors
        - Models focus on modifiable risk factors: Salt Intake, Stress, Sleep, BMI, Exercise, Smoking, Family History
        - **80:20 train-test split** with stratification
        - **5-Fold Stratified Cross-Validation** on training data only
        
        #### Methodology:
        
        1. **Data Validation**: Check uploaded data format and quality
        2. **Statistical Analysis**: Comprehensive EDA of all variables including categorical analysis
        3. **Machine Learning**: Multiple models with cross-validation
           - Logistic Regression (with Standard Scaling)
           - Random Forest Classifier
           - XGBoost Classifier (if available)
        4. **Risk Calculation**: ML-based prediction using the best performing model
        
        #### Learning Curves:
        
        - Learning curves start from as low as 5% of your dataset
        - Shows how model performance improves with more data
        - Helps detect overfitting (gap between training and validation curves)
        - Includes confidence bands showing variability across CV folds
        
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
        
        1. **Logistic Regression** (Scaled features)
        2. **Random Forest Classifier**
        3. **XGBoost Classifier** (if available)
        
        #### Model Validation:
        
        - 5-Fold Stratified Cross-Validation
        - Learning Curves (15 points from 5% to 100% data)
        - Confusion Matrices for each model
        - Test set evaluation (20% holdout)
        
        #### Key Metrics:
        
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - AUC-ROC
        - Cross-Validation Accuracy
        
        ---
        
        #### Version Information
        
        - Current Version: 8.0.0
        - Last Updated: April 2026
        - Data Source: User-uploaded CSV files
        - Minimum Data: 50 records recommended
        
        #### Privacy Note
        
        - All data processing happens locally in your browser
        - No data is sent to external servers
        - You maintain full control of your data
        """)
    
    st.markdown("---")
    st.markdown("### Resources and Downloads")
    
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
    st.markdown("Version 8.0 | April 2026")

with footer_col2:
    st.markdown("**No Data Leakage**")
    st.markdown("BP_History & Medication Excluded")

with footer_col3:
    st.markdown("**Built with Streamlit**")
    st.markdown("Data Science Project")
