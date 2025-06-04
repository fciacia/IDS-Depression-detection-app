# depression_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page settings
st.set_page_config(
    page_title="Depression Detection App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #e3f2fd;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Card-like containers */
    .content-container {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        margin: 1rem 0;
        border-radius: 2rem;
        background: linear-gradient(45deg, #2196F3, #64B5F6);
        color: white;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        background: linear-gradient(45deg, #1976D2, #2196F3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #81C784);
    }
    
    /* Alert boxes */
    .error-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #ffebee;
        border: none;
        box-shadow: 0 2px 4px rgba(239, 83, 80, 0.2);
        margin: 1rem 0;
    }
    
    .success-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #e8f5e9;
        border: none;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #fff3e0;
        border: none;
        box-shadow: 0 2px 4px rgba(255, 152, 0, 0.2);
        margin: 1rem 0;
    }
    
    .info-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 1rem;
        background-color: #e3f2fd;
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.2);
    }
    
    /* Titles and Headers */
    .main-title {
        text-align: center;
        margin-bottom: 2rem;
        color: #1565C0;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-title {
        color: #1976D2;
        font-size: 1.8rem;
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #90CAF9;
    }
    
    /* Center content */
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    
    /* Form styling */
    .stSelectbox {
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 0.5rem;
        border: 2px solid #90CAF9;
    }
    
    .stSlider > div > div {
        background-color: #BBDEFB;
    }
    
    /* Markdown text */
    .markdown-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #37474F;
    }
    
    /* Custom list styling */
    .custom-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .custom-list li {
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .custom-list li:before {
        content: "•";
        color: #2196F3;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Statistics Cards */
    .stats-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 200px;
        border-left: 4px solid #2196F3;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #546E7A;
        font-size: 1rem;
    }
    
    /* Story Cards */
    .story-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #FF9800;
    }
    
    .story-title {
        color: #FF9800;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .quote-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-style: italic;
    }
    
    .quote-text {
        color: #1565C0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .quote-author {
        color: #546E7A;
        font-size: 0.9rem;
        margin-top: 1rem;
        text-align: right;
    }
    
    /* Back Button */
    .back-button {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 1000;
    }
    
    .back-button button {
        background: linear-gradient(45deg, #90CAF9, #64B5F6) !important;
        border-radius: 2rem !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 1rem !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        width: auto !important;
        margin: 0 !important;
    }
    
    .back-button button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        background: linear-gradient(45deg, #64B5F6, #42A5F5) !important;
    }
    
    /* Main CTA Button */
    .main-cta {
        text-align: center;
        padding: 1rem;
        margin: 2rem auto;
        max-width: 500px;
    }
    
    .main-cta button {
        background: linear-gradient(45deg, #2196F3, #1976D2) !important;
        padding: 1rem 3rem !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        border-radius: 3rem !important;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3) !important;
        transition: all 0.3s ease !important;
        border: none !important;
        color: white !important;
        width: auto !important;
    }
    
    .main-cta button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4) !important;
        background: linear-gradient(45deg, #1976D2, #1565C0) !important;
    }
    
    .main-cta-description {
        color: #546E7A;
        font-size: 1.1rem;
        margin-top: 1rem;
        text-align: center;
    }
    
    /* Content Sections */
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .content-section {
        margin-bottom: 2rem;
    }
    
    .info-section, .process-section {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-section .section-title,
    .process-section .section-title {
        color: #1976D2;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #90CAF9;
    }
    
    .custom-list {
        list-style-type: none;
        padding-left: 0;
        margin: 1rem 0;
    }
    
    .custom-list li {
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
        color: #37474F;
        font-size: 1.1rem;
    }
    
    .custom-list li:before {
        content: "•";
        color: #2196F3;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    ol.custom-list {
        counter-reset: item;
    }
    
    ol.custom-list li {
        counter-increment: item;
    }
    
    ol.custom-list li:before {
        content: counter(item) ".";
        color: #2196F3;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
REQUIRED_FILES = {
    "trained_model.joblib": "Main prediction model",
    "scaler.joblib": "Data scaling preprocessor",
    "feature_names.txt": "Feature names configuration"
}

VALID_RANGES = {
    'academic_pressure': (1, 5),
    'study_satisfaction': (1, 5),
    'work_study_hours': (0, 24),
    'financial_stress': (1, 5),
    'cgpa': (0.0, 4.0)
}

@st.cache_resource
def load_model_and_resources() -> Tuple[Optional[Any], Optional[Any], Optional[list], Optional[str]]:
    """
    Load model and related resources with comprehensive error handling.
    
    Returns:
        Tuple containing:
        - model: Trained model object or None if loading fails
        - scaler: Data scaler object or None if loading fails
        - feature_names: List of feature names or None if loading fails
        - error: Error message string or None if loading succeeds
    """
    try:
        # Verify all required files exist
        missing_files = []
        for file_path, description in REQUIRED_FILES.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_path} ({description})")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing_files))
        
        # Load and validate model
        model = joblib.load("trained_model.joblib")
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise ValueError("Invalid model: missing required methods")
        
        # Load and validate scaler
        scaler = joblib.load("scaler.joblib")
        if not hasattr(scaler, 'transform'):
            raise ValueError("Invalid scaler: missing transform method")
        
        # Load and validate feature names
        with open('feature_names.txt', 'r') as f:
            feature_names = f.read().splitlines()
        
        if not feature_names:
            raise ValueError("Feature names file is empty")
        
        # Log successful loading
        logger.info("Successfully loaded model and resources")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        return model, scaler, feature_names, None
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        return None, None, None, str(e)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return None, None, None, str(e)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None, None, None, f"An unexpected error occurred: {str(e)}"

def validate_input(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate user input data against defined ranges and rules.
    
    Args:
        data: Dictionary containing user input data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        for field, (min_val, max_val) in VALID_RANGES.items():
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)):
                    return False, f"{field} must be a number"
                if value < min_val or value > max_val:
                    return False, f"{field} must be between {min_val} and {max_val}"
        
        required_fields = [
            'gender', 'sleep_duration', 'dietary_habits', 'degree',
            'family_mental_history', 'suicidal_thoughts'
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return False, f"Missing required field: {field}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

# Load model and resources
model, scaler, feature_names, load_error = load_model_and_resources()

# Initialize session state for page flow
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Main content
if load_error:
    st.error(f"Error loading required files: {load_error}")
    st.stop()

# Page 1: Home
if st.session_state.page == "Home":
    st.markdown("<h1 class='main-title'>Depression Detection App</h1>", unsafe_allow_html=True)
    
    # Main CTA Button at the top
    st.markdown("""
        <div class='main-cta'>
            <div class='main-cta-description'>
                Take the first step towards better mental health
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Analysis", use_container_width=True):
            st.session_state.page = "Input Form"
            st.rerun()
    
    # Statistics Cards
    st.markdown("""
        <div class='content-container'>
            <h2 class='section-title'>Mental Health Statistics</h2>
            <div class='stats-container'>
                <div class='stat-card'>
                    <div class='stat-number'>1 in 3</div>
                    <div class='stat-label'>College students experience significant depression symptoms</div>
                </div>
                <div class='stat-card'>
                    <div class='stat-number'>67%</div>
                    <div class='stat-label'>Students don't seek help due to stigma</div>
                </div>
                <div class='stat-card'>
                    <div class='stat-number'>80%</div>
                    <div class='stat-label'>Depression is treatable with proper support</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='main-content'>
            <div class='content-section'>
                <div class='story-card'>
                    <div class='story-title'>Why This Matters</div>
                    <p>Depression among students is more common than you might think. Academic pressure, 
                    social adjustments, and personal challenges can all contribute to mental health concerns. 
                    Early detection and support can make a significant difference in managing these challenges.</p>
                </div>
            </div>

            <div class='content-section'>
                <div class='quote-card'>
                    <div class='quote-text'>
                        "Mental health is not a destination, but a process. It's about how you drive, 
                        not where you're going."
                    </div>
                    <div class='quote-author'>- Noam Shpancer, PhD</div>
                </div>
            </div>

            <div class='content-section'>
                <div class='info-section'>
                    <h3 class='section-title'>Key Areas We Assess</h3>
                    <ul class='custom-list'>
                        <li>Academic pressure and satisfaction</li>
                        <li>Sleep patterns and dietary habits</li>
                        <li>Work-study balance</li>
                        <li>Financial stress</li>
                        <li>Personal and family history</li>
                    </ul>
                </div>
            </div>

            <div class='content-section'>
                <div class='process-section'>
                    <h3 class='section-title'>How it works:</h3>
                    <ol class='custom-list'>
                        <li>Fill out a simple questionnaire about your current situation</li>
                        <li>Our AI model analyzes your responses</li>
                        <li>Receive personalized feedback and recommendations</li>
                    </ol>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Page 2: Input Form
elif st.session_state.page == "Input Form":
    # Add back button at the top
    st.markdown("<div class='back-button'>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.page = "Home"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>Depression Risk Assessment Form</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    with st.form("depression_assessment"):
        st.markdown("<h3 class='section-title'>Personal Information</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            degree = st.selectbox("Degree Level", ["Pre-U", "Diploma", "Undergraduate", "Postgraduate", "PhD"])
            cgpa = st.slider("CGPA (0-4 scale)", 0.0, 4.0, 2.0, 0.1,
                           help="Your current Cumulative Grade Point Average")
        
        with col2:
            sleep_duration = st.selectbox("Average Sleep Duration", [
                "Less than 5 hours",
                "5-6 hours",
                "7-8 hours",
                "More than 8 hours"
            ])
            dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"],
                                       help="How would you rate your overall eating habits?")
        
        st.markdown("<h3 class='section-title'>Academic and Work</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            academic_pressure = st.slider("Academic Pressure", 1, 5,
                                       help="1 = Very Low, 5 = Very High")
            study_satisfaction = st.slider("Study Satisfaction", 1, 5,
                                        help="1 = Very Dissatisfied, 5 = Very Satisfied")
        
        with col2:
            work_study_hours = st.number_input("Daily Work/Study Hours", 0, 24, 8,
                                             help="Average hours spent on work/study per day")
            financial_stress = st.slider("Financial Stress Level", 1, 5,
                                      help="1 = No Stress, 5 = Severe Stress")
        
        st.markdown("<h3 class='section-title'>Mental Health History</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            family_mental_history = st.selectbox("Family History of Mental Health Issues",
                                               ["No", "Yes"],
                                               help="Has anyone in your immediate family been diagnosed with mental health issues?")
        
        with col2:
            suicidal_thoughts = st.selectbox("Have you had thoughts of self-harm?",
                                           ["No", "Yes"],
                                           help="This information is confidential and helps in risk assessment")
        
        st.markdown("""
            <div class='warning-box'>
                <h4>⚠️ Privacy Notice</h4>
                <p>Your responses are confidential and used only for assessment purposes. 
                If you're experiencing severe symptoms, please seek immediate professional help.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='center-content'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit Assessment")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submitted:
            with st.spinner("Processing your responses..."):
                try:
                    # Encode categorical variables
                    gender_encoded = 0 if gender == "Female" else 1
                    sleep_mapping = {
                        "Less than 5 hours": 1,
                        "5-6 hours": 2,
                        "7-8 hours": 3,
                        "More than 8 hours": 4
                    }
                    dietary_mapping = {
                        "Unhealthy": 1,
                        "Moderate": 2,
                        "Healthy": 3
                    }
                    degree_mapping = {
                        "Pre-U": 0,
                        "Diploma": 1,
                        "Undergraduate": 2,
                        "Postgraduate": 3,
                        "PhD": 4
                    }

                    # Calculate CGPA bin
                    cgpa_scaled = cgpa
                    bin_edges = np.arange(0.0, 4.01, 0.1)
                    cgpa_bin = pd.cut([cgpa_scaled], bins=bin_edges, labels=False, include_lowest=True)[0]
                    
                    # Create input data dictionary
                    input_data = {
                        'gender': gender_encoded,
                        'academic_pressure': academic_pressure,
                        'study_satisfaction': study_satisfaction,
                        'sleep_duration': sleep_mapping[sleep_duration],
                        'dietary_habits': dietary_mapping[dietary_habits],
                        'degree': degree_mapping[degree],
                        'suicidal_thoughts': 1 if suicidal_thoughts == "Yes" else 0,
                        'work_study_hours': work_study_hours,
                        'financial_stress': financial_stress,
                        'family_mental_history': 1 if family_mental_history == "Yes" else 0,
                        'age_bin': 0,  # Default age bin
                        'cgpa_scaled': cgpa_scaled,
                        'cgpa_bin': cgpa_bin
                    }
                    
                    # Validate input data
                    is_valid, error_message = validate_input(input_data)
                    if not is_valid:
                        st.error(f"Invalid input: {error_message}")
                        st.stop()
                    
                    # Create DataFrame and validate features
                    input_df = pd.DataFrame([input_data])
                    if set(feature_names[:-1]) != set(input_df.columns):
                        missing_features = set(feature_names[:-1]) - set(input_df.columns)
                        extra_features = set(input_df.columns) - set(feature_names[:-1])
                        error_msg = []
                        if missing_features:
                            error_msg.append(f"Missing features: {', '.join(missing_features)}")
                        if extra_features:
                            error_msg.append(f"Extra features: {', '.join(extra_features)}")
                        raise ValueError("\n".join(error_msg))
                    
                    # Reorder columns to match feature_names
                    input_df = input_df[feature_names[:-1]]
                    
                    # Store in session state
                    st.session_state["input_data"] = input_df
                    st.session_state["raw_input"] = {
                        'gender': gender,
                        'degree': degree,
                        'cgpa': cgpa,
                        'sleep_duration': sleep_duration,
                        'dietary_habits': dietary_habits,
                        'academic_pressure': academic_pressure,
                        'study_satisfaction': study_satisfaction,
                        'work_study_hours': work_study_hours,
                        'financial_stress': financial_stress,
                        'family_mental_history': family_mental_history,
                        'suicidal_thoughts': suicidal_thoughts
                    }
                    
                    # Navigate to prediction page
                    st.session_state.page = "Prediction"
                    st.rerun()
                    
                except Exception as e:
                    st.error("An error occurred while processing your input.")
                    st.error(f"Error details: {str(e)}")
                    logger.error(f"Form processing error: {str(e)}")
                    st.write("Please try again or contact support if the error persists.")
    st.markdown("</div>", unsafe_allow_html=True)

# Page 3: Prediction
elif st.session_state.page == "Prediction":
    # Add back button at the top
    st.markdown("<div class='back-button'>", unsafe_allow_html=True)
    if st.button("← Back to Form"):
        st.session_state.page = "Input Form"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>Assessment Results</h1>", unsafe_allow_html=True)

    if "input_data" not in st.session_state:
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.warning("Please complete the assessment form first.")
        st.markdown("<div class='center-content'>", unsafe_allow_html=True)
        if st.button("Go to Assessment Form"):
            st.session_state.page = "Input Form"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Analyzing your responses..."):
                # Get input data and scale it
                input_df = st.session_state["input_data"]
                input_scaled = scaler.transform(input_df)
                
                # Make prediction with error handling
                try:
                    prediction = model.predict(input_scaled)[0]
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    confidence = prediction_proba[prediction]
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
                    st.error("Error making prediction. Please try again.")
                    st.stop()

            st.markdown("<div class='content-container'>", unsafe_allow_html=True)
            # Display results in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<h3 class='section-title'>Risk Assessment</h3>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown("""
                        <div class='error-box'>
                            <h4>⚠️ Depression Risk Detected</h4>
                            <p>Based on your responses, you may be experiencing signs of depression.</p>
                            <p><strong>Important:</strong> This is not a diagnosis. Please consult a mental health professional for proper evaluation.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='success-box'>
                            <h4>✅ Low Risk Detected</h4>
                            <p>Your responses suggest a lower risk of depression.</p>
                            <p>However, mental health can change over time. Stay mindful of your well-being.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<h3 class='section-title'>Confidence Level</h3>", unsafe_allow_html=True)
                st.progress(confidence)
                st.write(f"Model confidence: {confidence:.1%}")
                
                st.markdown("<h3 class='section-title'>Key Contributing Factors</h3>", unsafe_allow_html=True)
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                feature_importance = pd.DataFrame({
                    'Factor': feature_names[:-1],
                    'Value': input_df.iloc[0].values,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.barh(feature_importance['Factor'][:5], feature_importance['Importance'][:5])
                plt.title("Top 5 Most Influential Factors")
                plt.xlabel("Relative Importance")
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.checkbox("Show Your Responses"):
                    st.markdown("<h3 class='section-title'>Your Responses</h3>", unsafe_allow_html=True)
                    raw_input = st.session_state.get("raw_input", {})
                    for key, value in raw_input.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                st.markdown("<h3 class='section-title'>Recommendations</h3>", unsafe_allow_html=True)
                if prediction == 1:
                    st.error("""
                    #### Immediate Actions:
                    1. **Seek Professional Help**
                       - Contact a mental health professional
                       - Visit your university counseling center
                       - Schedule a doctor's appointment
                    
                    2. **Emergency Resources**
                       - National Crisis Hotline: 988
                       - Crisis Text Line: Text HOME to 741741
                       
                    3. **Support System**
                       - Talk to trusted friends or family
                       - Join support groups
                       - Connect with student services
                    """)
                    
                    st.markdown("""
                        <div class='warning-box'>
                            <h4>🚨 Important Note</h4>
                            <p>If you're having thoughts of self-harm, please seek immediate help:</p>
                            <ul>
                                <li>Call emergency services</li>
                                <li>Go to the nearest emergency room</li>
                                <li>Contact a crisis hotline</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("""
                    #### Maintain Well-being:
                    1. **Mental Health**
                       - Practice stress management
                       - Maintain work-life balance
                       - Regular exercise
                    
                    2. **Academic Success**
                       - Use university resources
                       - Join study groups
                       - Take regular breaks
                       
                    3. **Social Connection**
                       - Stay connected with friends
                       - Participate in activities
                       - Build support network
                    """)
                    
                    st.info("""
                        #### Prevention Tips:
                        - Maintain regular sleep schedule
                        - Practice healthy eating habits
                        - Exercise regularly
                        - Stay connected with others
                        - Seek help early if needed
                    """)
            
            # Navigation - single button to start over
            st.markdown("---")
            st.markdown("<div class='center-content'>", unsafe_allow_html=True)
            if st.button("Start New Assessment"):
                st.session_state.pop("input_data", None)
                st.session_state.pop("raw_input", None)
                st.session_state.page = "Home"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"Error in prediction page: {str(e)}")
            st.error("An error occurred while processing your results.")
            st.error(f"Error details: {str(e)}")
            st.markdown("<div class='center-content'>", unsafe_allow_html=True)
            if st.button("Try Again"):
                st.session_state.page = "Input Form"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
