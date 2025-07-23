import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import io
import zipfile
import pickle
import json
import matplotlib 

# Page configuration
st.set_page_config(
    page_title="ML Visual Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === THEME: Toggle dark/light mode ===
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# Set Plotly theme based on the selected mode
plotly_template = "plotly_dark" if dark_mode else "plotly_white"

# Custom CSS for a modern and elegant design
if dark_mode:
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS pour le mode sombre */
    :root {
        --primary-color: #818cf8;
        --secondary-color: #a78bfa;
        --accent-color: #22d3ee;
        --success-color: #34d399;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --background-gradient: linear-gradient(135deg, #4338ca 0%, #7c3aed 100%);
        --card-shadow: 0 10px 25px rgba(0,0,0,0.3);
        --border-radius: 12px;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border-color: #475569;
    }
    
    /* Force dark mode on all elements */
    .stApp, .main, .block-container {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Global text styling */
    * {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb, .css-17eq0hr, section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: var(--bg-secondary) !important;
    }
    
    /* Sidebar content */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
    }

    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    
    p, span, div, label, .stMarkdown {
        color: var(--text-secondary) !important;
    }

    /* Header principal avec gradient sombre */
    .main-header {
        background: var(--background-gradient) !important;
        color: white !important;
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: white !important;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
        color: white !important;
    }

    /* Welcome page specific styles */
    .welcome-hero {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        color: var(--text-primary) !important;
        padding: 3rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }
    
    .welcome-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(129, 140, 248, 0.1) 0%, rgba(167, 139, 250, 0.1) 100%);
        z-index: 1;
    }
    
    .welcome-hero > * {
        position: relative;
        z-index: 2;
    }
    
    .feature-card {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        border: 1px solid var(--border-color) !important;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        transition: width 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .feature-card:hover::before {
        width: 100%;
        opacity: 0.1;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        color: var(--text-primary) !important;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        border: 1px solid var(--border-color) !important;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: scale(1.05);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color) !important;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Cards modernes pour mode sombre */
    .modern-card {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        border: 1px solid var(--border-color) !important;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    }
    
    .modern-card h4 {
        color: var(--text-primary) !important;
        margin-bottom: 1rem;
    }
    
    .modern-card p, .modern-card li {
        color: var(--text-secondary) !important;
    }

    /* Métriques stylées pour mode sombre */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        color: var(--text-primary) !important;
        padding: 1rem;
        border-radius: var(--border-radius);
        text-align: center;
        min-width: 120px;
        border: 1px solid var(--border-color) !important;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color) !important;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-muted) !important;
        margin-top: 0.25rem;
    }

    /* Boutons modernes pour mode sombre */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(129, 140, 248, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(129, 140, 248, 0.4);
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%) !important;
    }

    /* DataFrame et tableaux - Updated selectors */
    .stDataFrame, .stDataFrame div, [data-testid="stDataFrame"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stDataFrame table, [data-testid="stDataFrame"] table {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .stDataFrame th, [data-testid="stDataFrame"] th {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    .stDataFrame td, [data-testid="stDataFrame"] td {
        background-color: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }

    /* st.metric widget - Updated selectors */
    [data-testid="stMetric"], .stMetric {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius) !important;
        padding: 1rem !important;
    }
    
    [data-testid="stMetric"] > div, .stMetric > div {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetric"] label, .stMetric label {
        color: var(--text-secondary) !important;
    }

    /* Formulaires : selects, sliders, inputs - Updated selectors */
    .stSelectbox > div > div, [data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: var(--border-radius);
    }
    
    .stSelectbox > div > div > div, [data-testid="stSelectbox"] > div > div > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .stMultiSelect > div > div, [data-testid="stMultiSelect"] > div > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
    }
    
    .stRadio > div, [data-testid="stRadio"] > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .stRadio label, [data-testid="stRadio"] label {
        color: var(--text-primary) !important;
    }
    
    .stCheckbox > label, [data-testid="stCheckbox"] > label {
        color: var(--text-primary) !important;
    }
    
    .stNumberInput > div > div > input, [data-testid="stNumberInput"] > div > div > input {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
    }
    
    .stTextInput > div > div > input, [data-testid="stTextInput"] > div > div > input {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
    }

    .stSlider > div, [data-testid="stSlider"] > div {
        color: var(--text-primary) !important;
    }
    
    .stSlider > div > div > div > div, [data-testid="stSlider"] > div > div > div > div {
        background-color: var(--primary-color) !important;
    }

    /* File uploader - Updated selectors */
    .stFileUploader > div, [data-testid="stFileUploader"] > div {
        background-color: var(--bg-secondary) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: var(--border-radius);
    }
    
    .stFileUploader label, [data-testid="stFileUploader"] label {
        color: var(--text-primary) !important;
    }

    /* Alertes stylées pour mode sombre - Updated selectors */
    .stAlert, [data-testid="stAlert"] {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--card-shadow);
    }
    
    .stSuccess, [data-testid="stSuccess"] {
        background-color: rgba(52, 211, 153, 0.1) !important;
        color: var(--success-color) !important;
        border-left: 4px solid var(--success-color) !important;
    }
    
    .stError, [data-testid="stError"] {
        background-color: rgba(248, 113, 113, 0.1) !important;
        color: var(--error-color) !important;
        border-left: 4px solid var(--error-color) !important;
    }
    
    .stWarning, [data-testid="stWarning"] {
        background-color: rgba(251, 191, 36, 0.1) !important;
        color: var(--warning-color) !important;
        border-left: 4px solid var(--warning-color) !important;
    }
    
    .stInfo, [data-testid="stInfo"] {
        background-color: rgba(34, 211, 238, 0.1) !important;
        color: var(--accent-color) !important;
        border-left: 4px solid var(--accent-color) !important;
    }

    /* Progress bars - Updated selectors */
    .stProgress > div > div, [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--accent-color) 100%) !important;
        border-radius: 10px;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .section-header h3 {
        color: var(--primary-color) !important;
        font-weight: 600;
        margin: 0;
    }

    /* Tabs - Updated selectors */
    .stTabs [data-baseweb="tab-list"], [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: var(--bg-secondary) !important;
        border-radius: var(--border-radius);
    }
    
    .stTabs [data-baseweb="tab"], [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--border-radius);
    }
    
    .stTabs [aria-selected="true"], [data-testid="stTabs"] [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Expander - Updated selectors */
    .streamlit-expanderHeader, [data-testid="stExpander"] .streamlit-expanderHeader {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius);
    }
    
    .streamlit-expanderContent, [data-testid="stExpander"] .streamlit-expanderContent {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
    }

    /* Tooltips - Updated selectors */
    div[data-testid="stTooltip"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Download button - Updated selectors */
    .stDownloadButton > button, [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, var(--success-color) 0%, var(--accent-color) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius);
        font-weight: 600;
    }

    /* Form - Updated selectors */
    .stForm, [data-testid="stForm"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--border-radius);
        padding: 1rem;
    }

    /* Toggle/Checkbox styling */
    .stToggle > div, [data-testid="stToggle"] > div {
        background-color: var(--bg-secondary) !important;
    }

    /* Column containers */
    .css-1r6slb0, .css-12w0qpk {
        background-color: var(--bg-primary) !important;
    }

    /* Additional comprehensive styling */
    .element-container, .stMarkdown, .stText {
        color: var(--text-primary) !important;
    }

    /* Plotly charts background */
    .js-plotly-plot {
        background-color: var(--bg-secondary) !important;
    }

    /* Animation pour les éléments */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1, .welcome-hero h1 {
            font-size: 2rem;
        }
        
        .metric-container, .stats-grid {
            flex-direction: column;
            gap: 1rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
        
        .feature-icon {
            font-size: 2.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Variables CSS pour la cohérence */
        :root {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --accent-color: #06b6d4;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --card-shadow: 0 10px 25px rgba(0,0,0,0.1);
            --border-radius: 12px;
        }
        
        /* Styles globaux */
        .main {
            font-family: 'Inter', sans-serif;
        }
        
        /* Header principal avec gradient */
        .main-header {
            background: var(--background-gradient);
            padding: 2rem;
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: var(--card-shadow);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        /* Welcome page specific styles */
        .welcome-hero {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 3rem;
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: var(--card-shadow);
            position: relative;
            overflow: hidden;
        }
        
        .welcome-hero::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            z-index: 1;
        }
        
        .welcome-hero > * {
            position: relative;
            z-index: 2;
        }
        
        .feature-card {
            background: white;
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            border: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            transition: width 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .feature-card:hover::before {
            width: 100%;
            opacity: 0.1;
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            display: block;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            text-align: center;
            border: 1px solid #cbd5e1;
            transition: transform 0.2s ease;
        }
        
        .stat-card:hover {
            transform: scale(1.05);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            display: block;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Cards modernes */
        .modern-card {
            background: white;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .modern-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        /* Métriques stylées */
        .metric-container {
            display: flex;
            justify-content: space-around;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 1rem;
            border-radius: var(--border-radius);
            text-align: center;
            min-width: 120px;
            border: 1px solid #cbd5e1;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.25rem;
        }
        
        /* Boutons modernes */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }
        
        /* Sidebar moderne */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        /* Alertes stylées */
        .stAlert {
            border-radius: var(--border-radius);
            border: none;
            box-shadow: var(--card-shadow);
        }
        
        /* Formulaires */
        .stSelectbox > div > div {
            border-radius: var(--border-radius);
            border: 2px solid #e5e7eb;
            transition: border-color 0.2s ease;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        /* Dataframes */
        .stDataFrame {
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--card-shadow);
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--accent-color) 100%);
            border-radius: 10px;
        }
        
        /* Section headers */
        .section-header {
            display: flex;
            align-items: center;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e5e7eb;
        }
        
        .section-header h3 {
            color: var(--primary-color);
            font-weight: 600;
            margin: 0;
        }
        
        /* Animation pour les éléments */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header h1, .welcome-hero h1 {
                font-size: 2rem;
            }
            
            .metric-container, .stats-grid {
                flex-direction: column;
                gap: 1rem;
            }
            
            .feature-card {
                padding: 1.5rem;
            }
            
            .feature-icon {
                font-size: 2.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# === MODERN SIDEBAR MENU ===
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #6366f1; font-weight: 700; margin: 0;">🧠 ML Studio</h2>
        <p style="color: #64748b; font-size: 0.875rem; margin: 0.5rem 0 0 0;">Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Welcome", "Data", "Dashboard", "Preprocessing", "Model", "Results", "Export"],
        icons=["house", "cloud-upload", "bar-chart-line", "tools", "cpu", "graph-up", "download"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#6d70ff", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "12px 16px",
                "border-radius": "8px",
                "color": "#374151",
                "font-weight": "500",
                "--hover-color": "#f3f4f6",
                "transition": "all 0.2s ease",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                "color": "white",
                "font-weight": "600",
                "box-shadow": "0 4px 12px rgba(99, 102, 241, 0.3)",
            },
        },
    )

# === PAGE 0: WELCOME ===
if selected == "Welcome":
    # Hero Section
    st.markdown("""
    <div class="welcome-hero fade-in">
        <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            🚀 ML Visual Studio
        </h1>
        <p style="font-size: 1.3rem; margin-bottom: 2rem; opacity: 0.8;">
            Next-generation artificial intelligence platform for data analysis, modeling, and visualization
        </p>
        <p style="font-size: 1rem; opacity: 0.7; max-width: 600px; margin: 0 auto;">
            Transform your data into intelligent insights with our comprehensive machine learning toolkit. 
            From data preprocessing to model deployment, everything you need in one modern interface.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown("### 📊 Platform Capabilities")
    
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <span class="stat-number">6</span>
            <div class="stat-label">Core Modules</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">CSV</span>
            <div class="stat-label">Data Formats</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">2</span>
            <div class="stat-label">ML Algorithms</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">100%</span>
            <div class="stat-label">Visual Interface</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card fade-in">
            <span class="feature-icon">📊</span>
            <h3 style="color: #374151; margin-bottom: 1rem;">Data Management</h3>
            <p style="color: #6b7280; line-height: 1.6;">
                Import, explore, and analyze your datasets with powerful visualization tools. 
                Support for CSV files with intelligent data type detection and missing value analysis.
            </p>
            <ul style="color: #6b7280; margin-top: 1rem;">
                <li>Automatic data profiling</li>
                <li>Interactive data preview</li>
                <li>Smart data type detection</li>
                <li>Missing value visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card fade-in">
            <span class="feature-icon">🛠️</span>
            <h3 style="color: #374151; margin-bottom: 1rem;">Advanced Preprocessing</h3>
            <p style="color: #6b7280; line-height: 1.6;">
                Prepare your data for machine learning with comprehensive preprocessing tools. 
                Handle missing values, encode categorical variables, and normalize features.
            </p>
            <ul style="color: #6b7280; margin-top: 1rem;">
                <li>Missing value imputation</li>
                <li>One-hot encoding</li>
                <li>Feature scaling & normalization</li>
                <li>Data transformation pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card fade-in">
            <span class="feature-icon">📈</span>
            <h3 style="color: #374151; margin-bottom: 1rem;">Model Performance</h3>
            <p style="color: #6b7280; line-height: 1.6;">
                Comprehensive model evaluation with detailed metrics, visualizations, and performance analysis. 
                Track training progress and validate model accuracy.
            </p>
            <ul style="color: #6b7280; margin-top: 1rem;">
                <li>Real-time training metrics</li>
                <li>Performance visualizations</li>
                <li>Feature importance analysis</li>
                <li>Interactive predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card fade-in">
            <span class="feature-icon">📋</span>
            <h3 style="color: #374151; margin-bottom: 1rem;">Interactive Dashboard</h3>
            <p style="color: #6b7280; line-height: 1.6;">
                Explore your data through beautiful, interactive visualizations. 
                Create charts, analyze correlations, and discover patterns in your dataset.
            </p>
            <ul style="color: #6b7280; margin-top: 1rem;">
                <li>Dynamic chart generation</li>
                <li>Correlation analysis</li>
                <li>Distribution visualization</li>
                <li>Statistical summaries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card fade-in">
            <span class="feature-icon">🧠</span>
            <h3 style="color: #374151; margin-bottom: 1rem;">Machine Learning</h3>
            <p style="color: #6b7280; line-height: 1.6;">
                Train powerful machine learning models with custom-built algorithms. 
                Support for both regression and classification tasks with hyperparameter tuning.
            </p>
            <ul style="color: #6b7280; margin-top: 1rem;">
                <li>Linear & Logistic Regression</li>
                <li>Custom hyperparameters</li>
                <li>Training visualization</li>
                <li>Model persistence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card fade-in">
            <span class="feature-icon">📦</span>
            <h3 style="color: #374151; margin-bottom: 1rem;">Export & Deployment</h3>
            <p style="color: #6b7280; line-height: 1.6;">
                Export your models, data, and results in multiple formats. 
                Generate comprehensive reports and download everything as a convenient ZIP archive.
            </p>
            <ul style="color: #6b7280; margin-top: 1rem;">
                <li>Model serialization (PKL)</li>
                <li>Data export (CSV)</li>
                <li>Results export (JSON)</li>
                <li>Automated reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 2rem;">
            <h2 style="color: #6366f1; margin-bottom: 1rem;">1️⃣</h2>
            <h4 style="color: #374151; margin-bottom: 1rem;">Upload Data</h4>
            <p style="color: #6b7280; font-size: 0.9rem;">
                Start by uploading your CSV dataset in the <strong>Data</strong> section. 
                Our platform will automatically analyze and profile your data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 2rem;">
            <h2 style="color: #8b5cf6; margin-bottom: 1rem;">2️⃣</h2>
            <h4 style="color: #374151; margin-bottom: 1rem;">Explore & Preprocess</h4>
            <p style="color: #6b7280; font-size: 0.9rem;">
                Use the <strong>Dashboard</strong> to explore your data, then apply preprocessing 
                techniques to prepare it for machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 2rem;">
            <h2 style="color: #06b6d4; margin-bottom: 1rem;">3️⃣</h2>
            <h4 style="color: #374151; margin-bottom: 1rem;">Train & Export</h4>
            <p style="color: #6b7280; font-size: 0.9rem;">
                Train your model in the <strong>Model</strong> section, analyze results, 
                and export everything for future use.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin: 2rem 0;">
        <h3 style="color: #374151; margin-bottom: 1rem;">Ready to Start Your ML Journey?</h3>
        <p style="color: #6b7280; margin-bottom: 2rem; max-width: 500px; margin-left: auto; margin-right: auto;">
            Begin by uploading your dataset and let our intelligent platform guide you through 
            the complete machine learning workflow.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Tips
    st.markdown("### 💡 Pro Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: #10b981; margin-bottom: 1rem;">🎯 Data Quality</h4>
            <ul style="color: #6b7280; font-size: 0.9rem;">
                <li>Ensure your CSV has clear column headers</li>
                <li>Check for consistent data types</li>
                <li>Remove or handle extreme outliers</li>
                <li>Verify data encoding (UTF-8 recommended)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: #f59e0b; margin-bottom: 1rem;">⚡ Performance</h4>
            <ul style="color: #6b7280; font-size: 0.9rem;">
                <li>Start with smaller datasets for testing</li>
                <li>Monitor training progress in real-time</li>
                <li>Export results regularly for backup</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# === HEADER PRINCIPAL (pour les autres pages) ===
if selected != "Welcome":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🚀 ML Visual Studio</h1>
        <p>Next-generation artificial intelligence platform for data analysis, modeling, and visualization</p>
    </div>
    """, unsafe_allow_html=True)

# === PAGE 1: DATA ===
if selected == "Data":
    st.markdown('<div class="section-header"><h3>📁 Dataset Import </h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: #374151; margin-bottom: 1rem;">📤 Load Dataset</h4>
            <p style="color: #6b7280; margin-bottom: 1rem;">
                Upload your CSV file to start the analysis. 
                Supported formats: CSV • Max size: 50MB • Encoding: UTF-8
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Select your CSV file",
            type=["csv"],
            help="Drag & drop your file here, or click to browse",
        )

    with col2:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: #374151; margin-bottom: 1rem;">💡 Tips</h4>
            <ul style="color: #6b7280; font-size: 0.875rem;">
                <li>Ensure the first row contains headers</li>
                <li>Avoid special characters in column names</li>
                <li>Check your file encoding</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df

            st.success("✅ File successfully imported!", icon="🎉")

            # Modern dataset preview
            st.markdown('<div class="section-header"><h3>🔍 Dataset Preview</h3></div>', unsafe_allow_html=True)

            # Metrics in modern cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.shape[0]:,}</div>
                    <div class="metric-label">📦 Rows</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.shape[1]}</div>
                    <div class="metric-label">🧱 Columns</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df.isnull().sum().sum()}</div>
                    <div class="metric-label">🔍 Missing values</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{memory_usage:.1f} MB</div>
                    <div class="metric-label">💾 Memory usage</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Data table with modern style
            st.markdown("**📋 Sample of the data**")
            st.dataframe(df.head(10), use_container_width=True, height=300)

            # Data type information
            st.markdown('<div class="section-header"><h3>🧬 Data Type Analysis</h3></div>', unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                dtypes_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Unique Values": [df[col].nunique() for col in df.columns],
                    "% Missing Values": [f"{(df[col].isnull().sum() / len(df) * 100):.1f}%" for col in df.columns]
                })
                st.dataframe(dtypes_df, use_container_width=True, height=300)

            with col2:
                # Donut chart of data types
                numeric_cols = len(df.select_dtypes(include=["int", "float"]).columns)
                categorical_cols = len(df.select_dtypes(include=["object", "category"]).columns)
                other_cols = df.shape[1] - numeric_cols - categorical_cols

                fig_types = px.pie(
                    values=[numeric_cols, categorical_cols, other_cols],
                    names = ["Numerical", "Categorical", "Others"],
                    hole=0.6,
                    color_discrete_sequence=["#6366f1", "#8b5cf6", "#06b6d4"],
                    template=plotly_template
                )
                fig_types.update_traces(textinfo='percent+label', textfont_size=12)
                fig_types.update_layout(
                    title="Distribution of Data Types",
                    showlegend=True,
                    height=300,
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_types, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #9ca3af;">📁 No file selected</h3>
            <p style="color: #6b7280;">Please upload a CSV file to start the analysis</p>
        </div>
        """, unsafe_allow_html=True)

# === PAGE 2: DASHBOARD ===
elif selected == "Dashboard":
    st.markdown('<div class="section-header"><h3>📊 Analytical Dashboard</h3></div>', unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #f59e0b;">⚠️ Missing data</h3>
            <p style="color: #6b7280;">Please upload a dataset first in the 'Data' tab</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state["df"]

        # Separate column types
        numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Descriptive stats with modern style
        st.markdown("### 📈 Descriptive Statistics")

        if numeric_cols:
            col1, col2 = st.columns([3, 1])

            with col1:
                stats_df = df[numeric_cols].describe().round(2)
                st.dataframe(stats_df.style.background_gradient(cmap="viridis", axis=1), use_container_width=True)

            with col2:
                if len(numeric_cols) >= 3:
                    means_normalized = (df[numeric_cols].mean() - df[numeric_cols].mean().min()) / (df[numeric_cols].mean().max() - df[numeric_cols].mean().min())

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=means_normalized.values,
                        theta=means_normalized.index,
                        fill='toself',
                        name='Normalized Means',
                        line_color='#6366f1'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])
                        ),
                        showlegend=False,
                        title="Variable Profile",
                        height=300,
                        template=plotly_template
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # Interactive visualizations
        st.markdown("### 🎨 Interactive Visualizations")

        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Trends"])

        with tab1:
            if numeric_cols:
                col1, col2 = st.columns([1, 3])

                with col1:
                    selected_cols = st.multiselect(
                        "Select variables:",
                        numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                    )

                    chart_type = st.radio(
                        "Chart type:",
                        ["Histogram", "Box Plot", "Violin Plot"]
                    )

                with col2:
                    if selected_cols:
                        for col in selected_cols:
                            if chart_type == "Histogram":
                                fig = px.histogram(
                                    df, x=col,
                                    title=f"Distribution of {col}",
                                    color_discrete_sequence=["#6366f1"],
                                    template=plotly_template
                                )
                            elif chart_type == "Box Plot":
                                fig = px.box(
                                    df, y=col,
                                    title=f"Box Plot of {col}",
                                    color_discrete_sequence=["#8b5cf6"],
                                    template=plotly_template
                                )
                            else:  # Violin Plot
                                fig = px.violin(
                                    df, y=col,
                                    title=f"Violin Plot of {col}",
                                    color_discrete_sequence=["#06b6d4"],
                                    template=plotly_template
                                )

                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()

                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Matrix",
                    template=plotly_template
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)

                strong_corr = corr_matrix.abs() > 0.5
                if strong_corr.any().any():
                    st.markdown("**🔥 Strong correlations detected (|r| > 0.5):**")
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:
                                col1, col2, col3 = st.columns([2, 1, 2])
                                with col1:
                                    st.write(f"**{corr_matrix.columns[i]}**")
                                with col2:
                                    st.metric("", f"{corr_val:.3f}")
                                with col3:
                                    st.write(f"**{corr_matrix.columns[j]}**")

        with tab3:
            if categorical_cols:
                selected_cat = st.selectbox("Categorical variable:", categorical_cols)

                if selected_cat:
                    cat_counts = df[selected_cat].value_counts().reset_index()
                    cat_counts.columns = [selected_cat, "Frequency"]

                    fig_cat = px.bar(
                        cat_counts.head(10),
                        x=selected_cat,
                        y="Frequency",
                        title=f"Top 10 - Distribution of {selected_cat}",
                        color="Frequency",
                        color_continuous_scale="viridis",
                        template=plotly_template
                    )
                    fig_cat.update_layout(height=400)
                    st.plotly_chart(fig_cat, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Unique values", df[selected_cat].nunique())
                    with col2:
                        st.metric("Mode", df[selected_cat].mode().iloc[0] if not df[selected_cat].mode().empty else "N/A")
                    with col3:
                        st.metric("Missing values", df[selected_cat].isnull().sum())


# === PAGE 3 : Preprocessing ===
elif selected == "Preprocessing":
    st.markdown('<div class="section-header"><h3>🛠️ Advanced Preprocessing</h3></div>', unsafe_allow_html=True)
    
    if "df" not in st.session_state:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #f59e0b;">⚠️ Missing data</h3>
            <p style="color: #6b7280;">Please upload a dataset first in the 'Data' tab</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state["df"].copy()
        
        # Progress bar for preprocessing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # === HANDLING MISSING VALUES ===
        st.markdown("### 🧼 Handling Missing Values")
        
        missing_df = df.isnull().sum()
        missing_cols = missing_df[missing_df > 0].index.tolist()
        
        if not missing_cols:
            st.success("🎉 No missing values detected in the dataset!")
        else:
            st.warning(f"⚠️ {len(missing_cols)} columns contain missing values")
            
            # Missing values visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                missing_data = pd.DataFrame({
                    'Column': missing_cols,
                    'Missing Values': [missing_df[col] for col in missing_cols],
                    'Percentage': [f"{(missing_df[col] / len(df) * 100):.1f}%" for col in missing_cols]
                })
                st.dataframe(missing_data, use_container_width=True)
            
            with col2:
                fig_missing = px.bar(
                    missing_data,
                    x='Missing Values',
                    y='Column',
                    orientation='h',
                    title="Missing Values by Column",
                    color='Missing Values',
                    color_continuous_scale="Reds",
                    template=plotly_template
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            
            # Treatment interface
            with st.expander("🔧 Treatment Configuration", expanded=True):
                method_map = {}
                
                for col in missing_cols:
                    st.markdown(f"**📊 Column: `{col}`** ({missing_df[col]} missing values)")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        method = st.selectbox(
                            f"Method for {col}:",
                            ["No action", "Drop column", "Mean", "Median", "Mode", "Custom value"],
                            key=f"method_{col}"
                        )
                    
                    with col2:
                        if method == "Custom value":
                            custom_value = st.text_input(f"Value for {col}:", key=f"custom_{col}")
                            method_map[col] = (method, custom_value)
                        else:
                            method_map[col] = (method, None)
                
                if st.button("🚀 Apply treatment", type="primary"):
                    progress_bar.progress(25)
                    status_text.text("Processing missing values...")
                    
                    for col, (strategy, custom) in method_map.items():
                        if strategy == "Drop column":
                            df.drop(columns=col, inplace=True)
                            st.info(f"🗑️ Column `{col}` dropped")
                        elif strategy == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].mean(), inplace=True)
                            st.success(f"✅ `{col}` filled with mean")
                        elif strategy == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].median(), inplace=True)
                            st.success(f"✅ `{col}` filled with median")
                        elif strategy == "Mode":
                            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else df[col].iloc[0]
                            df[col].fillna(mode_val, inplace=True)
                            st.success(f"✅ `{col}` filled with mode")
                        elif strategy == "Custom value":
                            df[col].fillna(custom, inplace=True)
                            st.success(f"✅ `{col}` filled with '{custom}'")
        
        progress_bar.progress(50)
        
        # === CATEGORICAL VARIABLE ENCODING ===
        st.markdown("---")
        st.markdown("### 🧠 Categorical Variable Encoding")
        
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if cat_cols:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_cat_cols = st.multiselect(
                    "📌 Columns to encode (One-Hot Encoding):",
                    cat_cols,
                    default=cat_cols[:3] if len(cat_cols) > 3 else cat_cols
                )
                
                if st.button("🔄 Apply Encoding", type="secondary"):
                    if selected_cat_cols:
                        progress_bar.progress(75)
                        status_text.text("Encoding categorical variables...")
                        
                        df = pd.get_dummies(df, columns=selected_cat_cols, drop_first=True)
                        for col in df.columns:
                            if df[col].dtype == 'bool':
                                df[col] = df[col].astype(int)

                        st.success(f"✅ Encoding applied to: {', '.join(selected_cat_cols)}")
            
            with col2:
                # Preview of encoding impact
                if selected_cat_cols:
                    total_new_cols = sum([df[col].nunique() - 1 for col in selected_cat_cols if col in df.columns])
                    st.metric("New columns created", total_new_cols)
                    st.metric("Current number of columns", len(df.columns))
        else:
            st.info("✅ No categorical variables detected")
        
        # === NORMALIZATION ===
        st.markdown("---")
        st.markdown("### 📏 Normalization of Numerical Variables")
        
        num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        
        if num_cols:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_num_cols = st.multiselect(
                    "🧮 Variables to normalize:",
                    num_cols,
                    default=num_cols[:5] if len(num_cols) > 5 else num_cols
                )
                
                scaler_choice = st.radio(
                    "Normalization method:",
                    ["None", "StandardScaler (Z-score)", "MinMaxScaler (0-1)"],
                    horizontal=True
                )
                
                if st.button("📊 Apply normalization", type="secondary"):
                    if scaler_choice != "None" and selected_num_cols:
                        progress_bar.progress(90)
                        status_text.text("Normalizing variables...")
                        
                        if scaler_choice == "StandardScaler (Z-score)":
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                        
                        df[selected_num_cols] = scaler.fit_transform(df[selected_num_cols])
                        st.success(f"✅ {scaler_choice} successfully applied")
            
            with col2:
                if selected_num_cols and len(selected_num_cols) > 0:
                    # Preview of before/after statistics
                    st.markdown("**📊 Overview of changes:**")
                    sample_col = selected_num_cols[0]
                    if sample_col in df.columns:
                        st.metric("Min", f"{df[sample_col].min():.3f}")
                        st.metric("Max", f"{df[sample_col].max():.3f}")
                        st.metric("Mean", f"{df[sample_col].mean():.3f}")
        
        progress_bar.progress(100)
        status_text.text("✅ Preprocessing completed!")
        
        # Save cleaned dataset
        st.session_state["df_cleaned"] = df
        
        # Final summary
        st.markdown("---")
        st.markdown("### 📋 Summary of the Preprocessed Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("🧱 Columns", df.shape[1])
        with col3:
            st.metric("🔍 Missing values", df.isnull().sum().sum())
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Size", f"{memory_usage:.1f} MB")
        
        # Final preview
        st.markdown("**🔍 Preview of the preprocessed dataset:**")
        st.dataframe(df.head(), use_container_width=True)


# === PAGE 4 : MODEL ===
elif selected == "Model":
    st.markdown('<div class="section-header"><h3>🧠 Train Your AI Model</h3></div>', unsafe_allow_html=True)
    
    if "df_cleaned" not in st.session_state:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #f59e0b;">⚠️ Preprocessing Required</h3>
            <p style="color: #6b7280;">Please preprocess your data first in the 'Preprocessing' section</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state["df_cleaned"]
        all_columns = df.columns.tolist()
        
        # Modern configuration interface
        st.markdown("### ⚙️ Model Configuration")
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="modern-card">
                    <h4 style="color: #374151; margin-bottom: 1rem;">🎯 Variable Selection</h4>
                </div>
                """, unsafe_allow_html=True)
                
                target = st.selectbox(
                    "🎯 Target Variable:",
                    all_columns,
                    help="The variable you want to predict"
                )
                
                available_features = [col for col in all_columns if col != target]
                features = st.multiselect(
                    "🧩 Explanatory Variables (features):",
                    available_features,
                    default=available_features[:5] if len(available_features) > 5 else available_features,
                    help="The variables used for prediction"
                )
                
                model_type = st.radio(
                    "🧠 Model Type:",
                    ["Regression", "Classification"],
                    horizontal=True,
                    help="Regression for continuous values, Classification for categories"
                )
            
            with col2:
                st.markdown("""
                <div class="modern-card">
                    <h4 style="color: #374151; margin-bottom: 1rem;">⚡ Training Parameters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                learning_rate = st.slider(
                    "📈 Learning Rate:",
                    0.001, 1.0, 0.01, step=0.001,
                    help="Controls how fast the model learns"
                )
                
                n_iter = st.slider(
                    "🔄 Number of Iterations:",
                    100, 5000, 1000, step=100,
                    help="Number of training steps"
                )
                
                test_size = st.slider(
                    "📊 Test Set Size (%):",
                    10, 50, 20,
                    help="Percentage of data reserved for testing"
                )
        
        # Validation and training
        if st.button("🚀 Launch Training", type="primary", use_container_width=True):
            if not features:
                st.error("❌ Please select at least one feature")
                st.stop()
            
            # Numeric type validation
            for col in features:
                # If it's boolean, convert to integer
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
                    st.info(f"🔄 Column '{col}' converted from boolean to integer")
                # If it's still not numeric, error
                elif not np.issubdtype(df[col].dtype, np.number):
                     st.error(f"❌ Column '{col}' is not numeric. Please apply encoding in the 'Preprocessing' section.")
                     st.stop()
            # Do the same for target variable
            if df[target].dtype == 'bool':
               df[target] = df[target].astype(int)
               st.info(f"🔄 Target variable '{target}' converted from boolean to integer")
            elif not np.issubdtype(df[target].dtype, np.number):
               st.error(f"❌ Target variable '{target}' is not numeric.")
               st.stop()
            
            # Training progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 Preparing data...")
                progress_bar.progress(10)
                
                X = df[features].values.astype(float)
                y = df[target].values.reshape(-1, 1).astype(float)
                
                # Data split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                
                # Normalization after split
                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0)
                X_train_norm = (X_train - X_mean) / X_std
                X_test_norm = (X_test - X_mean) / X_std
                
                progress_bar.progress(20)
                status_text.text("🧠 Initializing model...")
                
                # Initialization
                weights = np.random.normal(0, 0.01, (X_train_norm.shape[1], 1))  # Random initialization
                bias = 0.0
                m = X_train_norm.shape[0]
                history = []
                
                progress_bar.progress(30)
                status_text.text("🏃‍♂️ Training in progress...")
                
                # Training with progress updates
                for i in range(n_iter):
                    linear_model = np.dot(X_train_norm, weights) + bias
                    
                    if model_type == "Regression":
                        y_pred = linear_model
                        errors = y_pred - y_train
                        cost = (1 / (2 * m)) * np.sum(errors ** 2)
                        dw = (1 / m) * np.dot(X_train_norm.T, errors)
                        db = (1 / m) * np.sum(errors)
                    else:
                        y_pred = 1 / (1 + np.exp(-linear_model))
                        errors = y_pred - y_train
                        cost = - (1 / m) * np.sum(y_train * np.log(y_pred + 1e-8) + (1 - y_train) * np.log(1 - y_pred + 1e-8))
                        dw = (1 / m) * np.dot(X_train_norm.T, errors)
                        db = (1 / m) * np.sum(errors)
                    
                    weights -= learning_rate * dw
                    bias -= learning_rate * db
                    history.append(cost)
                    
                    # Progress update
                    if i % (n_iter // 50) == 0:
                        progress = 30 + int((i / n_iter) * 60)
                        progress_bar.progress(progress)
                
                progress_bar.progress(90)
                status_text.text("📊 Evaluating model...")
                
                # Final prediction
                linear_test = np.dot(X_test_norm, weights) + bias
                if model_type == "Regression":
                    y_test_pred = linear_test
                else:
                    y_test_pred = (1 / (1 + np.exp(-linear_test)) >= 0.5).astype(int)
                
                # Save to session state
                st.session_state["trained_model"] = {
                    "weights": weights, 
                    "bias": bias, 
                    "X_mean": X_mean, 
                    "X_std": X_std
                }
                st.session_state["X_test"] = X_test_norm
                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_test_pred
                st.session_state["history"] = history
                st.session_state["model_config"] = {
                    "type": model_type,
                    "features": features,
                    "target": target,
                    "learning_rate": learning_rate,
                    "n_iter": n_iter,
                    "test_size": test_size
                }
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed successfully!")
                
                st.success("🎉 Model trained successfully! Check the results in the 'Results' tab")
                
                # Quick performance overview
                if model_type == "Regression":
                    mse = mean_squared_error(y_test, y_test_pred)
                    r2 = r2_score(y_test, y_test_pred)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📉 MSE", f"{mse:.4f}")
                    with col2:
                        st.metric("📈 R²", f"{r2:.4f}")
                else:
                    acc = accuracy_score(y_test, y_test_pred)
                    st.metric("✅ Accuracy", f"{acc:.2%}")
                
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
                progress_bar.progress(0)
                status_text.text("")



# === PAGE 5 : RESULTS ===
elif selected == "Results":
    st.markdown('<div class="section-header"><h3>📈 Results Analysis</h3></div>', unsafe_allow_html=True)
    
    if "trained_model" not in st.session_state or "history" not in st.session_state:
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #f59e0b;">⚠️ No Trained Model</h3>
            <p style="color: #6b7280;">Please train a model first in the 'Model' tab</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        model = st.session_state["trained_model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        model_type = st.session_state["model_config"]["type"]
        cost_history = st.session_state["history"]
        features = st.session_state["model_config"]["features"]
        X_mean = model["X_mean"]
        X_std = model["X_std"]
        
        # Header with model information
        st.markdown(f"""
        <div class="modern-card">
            <h3 style="color: #6366f1; margin-bottom: 1rem;">🤖 Model: {model_type}</h3>
            <p style="color: #6b7280; margin: 0;">
                <strong>Features:</strong> {', '.join(features[:3])}{'...' if len(features) > 3 else ''} 
                | <strong>Target:</strong> {st.session_state["model_config"]["target"]}
                | <strong>Test samples:</strong> {len(y_test)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # === PERFORMANCE METRICS ===
        st.markdown("### 📊 Performance Metrics")
        
        if model_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{mse:.4f}</div>
                    <div class="metric-label">📉 MSE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{rmse:.4f}</div>
                    <div class="metric-label">📐 RMSE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{r2:.4f}</div>
                    <div class="metric-label">📈 R²</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{mae:.4f}</div>
                    <div class="metric-label">📏 MAE</div>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Classification
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{acc:.2%}</div>
                    <div class="metric-label">✅ Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{precision:.2%}</div>
                    <div class="metric-label">🎯 Precision</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{recall:.2%}</div>
                    <div class="metric-label">📢 Recall</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{f1:.2%}</div>
                    <div class="metric-label">🏅 F1-Score</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # === VISUALIZATIONS ===
        tab1, tab2, tab3, tab4 = st.tabs(["📉 Convergence", "🔍 Predictions", "🧠 Features", "🎯 Matrix"])
        
        with tab1:
            # Modern convergence curve
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                y=cost_history,
                mode='lines',
                line=dict(color='#6366f1', width=3),
                name='Cost',
                fill='tonexty',
                fillcolor='rgba(99, 102, 241, 0.1)'
            ))
            fig_cost.update_layout(
                title="📉 Cost Evolution During Training",
                xaxis_title="Iteration",
                yaxis_title="Cost",
                template=plotly_template,
                height=400,
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            
            # Convergence statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Cost", f"{cost_history[0]:.6f}")
            with col2:
                st.metric("Final Cost", f"{cost_history[-1]:.6f}")
            with col3:
                improvement = ((cost_history[0] - cost_history[-1]) / cost_history[0]) * 100
                st.metric("Improvement", f"{improvement:.2f}%")
        
        with tab2:
            # Predictions vs actual graph
            fig_pred = go.Figure()
            
            # Perfect prediction reference line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction'
            ))
            
            # Prediction points
            fig_pred.add_trace(go.Scatter(
                x=y_test.flatten(),
                y=y_pred.flatten(),
                mode='markers',
                marker=dict(
                    color='#6366f1',
                    size=8,
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                name='Predictions'
            ))
            
            fig_pred.update_layout(
                title="🔍 Predictions vs Actual Values",
                xaxis_title="Actual Values",
                yaxis_title="Predictions",
                template=plotly_template,
                height=400,
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with tab3:
            # Feature importance
            weights = model["weights"]
            importance_df = pd.DataFrame({
                'Feature': features,
                'Weight': weights.flatten(),
                'Importance': np.abs(weights.flatten())
            }).sort_values('Importance', ascending=True)
            
            fig_weights = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="🧠 Feature Importance (Absolute Weight Values)",
                color='Importance',
                color_continuous_scale="viridis",
                template=plotly_template
            )
            fig_weights.update_layout(height=400)
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Detailed table
            st.markdown("**📊 Weight Details:**")
            st.dataframe(importance_df.sort_values('Importance', ascending=False), use_container_width=True)
        
        with tab4:
            if model_type == "Classification":
                # Modern confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="🎯 Confusion Matrix",
                    template=plotly_template
                )
                fig_cm.update_layout(
                    xaxis_title="Predictions",
                    yaxis_title="Actual Values",
                    height=400
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                # For regression, show error distribution
                errors = y_test.flatten() - y_pred.flatten()
                
                fig_errors = px.histogram(
                    x=errors,
                    nbins=30,
                    title="📊 Prediction Error Distribution",
                    color_discrete_sequence=["#6366f1"],
                    template=plotly_template
                )
                fig_errors.update_layout(
                    xaxis_title="Error (Actual - Predicted)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_errors, use_container_width=True)
        
        # === INTERACTIVE PREDICTION ===
        st.markdown("---")
        st.markdown("### 🔮 Interactive Prediction")
        
        with st.expander("✨ Test your model with new data", expanded=True):
            with st.form("prediction_form"):
                st.markdown("**📝 Enter feature values:**")
                
                # Organize inputs in columns
                n_cols = min(3, len(features))
                cols = st.columns(n_cols)
                user_input = []
                
                for i, feature in enumerate(features):
                    col = cols[i % n_cols]
                    with col:
                        value = st.number_input(
                            f"🔢 {feature}",
                            key=f"input_{feature}",
                            help=f"Value for feature '{feature}'"
                        )
                        user_input.append(value)
                
                submitted = st.form_submit_button("🚀 Make Prediction", type="primary", use_container_width=True)
            
            if submitted:
                try:
                    input_array = np.array(user_input).reshape(1, -1)
                    input_array_norm = (input_array - X_mean) / X_std
                    
                    if model_type == "Regression":
                        prediction = np.dot(input_array_norm, weights) + model["bias"]
                        
                        st.markdown("""
                        <div class="modern-card" style="text-align: center;">
                            <h3 style="color: #10b981; margin-bottom: 1rem;">💡 Prediction Result</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card" style="padding: 2rem;">
                                <div style="font-size: 2rem; font-weight: 700; color: #6366f1;">{prediction[0][0]:.4f}</div>
                                <div style="font-size: 1rem; color: #6b7280;">Predicted Value</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:  # Classification
                        z = np.dot(input_array_norm, weights) + model["bias"]
                        proba = 1 / (1 + np.exp(-z))
                        classe = 1 if proba >= 0.5 else 0
                        
                        st.markdown("""
                        <div class="modern-card" style="text-align: center;">
                            <h3 style="color: #10b981; margin-bottom: 1rem;">🎯 Classification Result</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card" style="padding: 2rem;">
                                <div style="font-size: 2rem; font-weight: 700; color: #6366f1;">{classe}</div>
                                <div style="font-size: 1rem; color: #6b7280;">Predicted Class</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card" style="padding: 2rem;">
                                <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6;">{proba[0][0]:.2%}</div>
                                <div style="font-size: 1rem; color: #6b7280;">Probability</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")


# === PAGE 6 : EXPORT ===
elif selected == "Export":
    st.markdown('<div class="section-header"><h3>📦 Results Export</h3></div>', unsafe_allow_html=True)
    
    # Check available data
    has_data = "df" in st.session_state
    has_cleaned_data = "df_cleaned" in st.session_state
    has_model = "trained_model" in st.session_state
    
    # Data status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅" if has_data else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status}</div>
            <div class="metric-label">Raw data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅" if has_cleaned_data else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status}</div>
            <div class="metric-label">Preprocessed data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = "✅" if has_model else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status}</div>
            <div class="metric-label">Trained model</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not any([has_data, has_cleaned_data, has_model]):
        st.markdown("""
        <div class="modern-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #f59e0b;">⚠️ No data to export</h3>
            <p style="color: #6b7280;">Please first import and process data</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Export options
        st.markdown("### 📋 Export Options")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            export_options = []
            
            if has_data:
                export_raw = st.checkbox("📊 Raw data (CSV)", value=True)
                if export_raw:
                    export_options.append("raw_data")
            
            if has_cleaned_data:
                export_cleaned = st.checkbox("🧹 Preprocessed data (CSV)", value=True)
                if export_cleaned:
                    export_options.append("cleaned_data")
            
            if has_model:
                export_model = st.checkbox("🧠 Trained model (PKL)", value=True)
                if export_model:
                    export_options.append("model")
                
                export_results = st.checkbox("📈 Results and metrics (JSON)", value=True)
                if export_results:
                    export_options.append("results")
            
            include_report = st.checkbox("📄 Analysis report (TXT)", value=True)
            if include_report:
                export_options.append("report")
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h4 style="color: #374151; margin-bottom: 1rem;">📦 Archive Format</h4>
                <p style="color: #6b7280; font-size: 0.875rem;">
                    Export generates a ZIP file containing all selected elements.
                </p>
                <ul style="color: #6b7280; font-size: 0.875rem;">
                    <li>CSV: Tabular data</li>
                    <li>PKL: Python model</li>
                    <li>JSON: Structured metrics</li>
                    <li>TXT: Readable report</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Export button
        if st.button("🚀 Generate Archive", type="primary", use_container_width=True):
            if not export_options:
                st.warning("⚠️ Please select at least one element to export")
            else:
                try:
                    # Create ZIP archive in memory
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        
                        # Export raw data
                        if "raw_data" in export_options and has_data:
                            csv_buffer = io.StringIO()
                            st.session_state["df"].to_csv(csv_buffer, index=False)
                            zip_file.writestr("raw_data.csv", csv_buffer.getvalue())
                        
                        # Export preprocessed data
                        if "cleaned_data" in export_options and has_cleaned_data:
                            csv_buffer = io.StringIO()
                            st.session_state["df_cleaned"].to_csv(csv_buffer, index=False)
                            zip_file.writestr("preprocessed_data.csv", csv_buffer.getvalue())
                        
                        # Export model
                        if "model" in export_options and has_model:
                            model_buffer = io.BytesIO()
                            pickle.dump(st.session_state["trained_model"], model_buffer)
                            zip_file.writestr("trained_model.pkl", model_buffer.getvalue())
                        
                        # Export results
                        if "results" in export_options and has_model:
                            results = {
                                "model_config": st.session_state["model_config"],
                                "training_history": st.session_state["history"]
                            }
                            
                            # Add metrics
                            if st.session_state["model_config"]["type"] == "Regression":
                                y_test = st.session_state["y_test"]
                                y_pred = st.session_state["y_pred"]
                                results["metrics"] = {
                                    "mse": float(mean_squared_error(y_test, y_pred)),
                                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                                    "r2": float(r2_score(y_test, y_pred)),
                                    "mae": float(np.mean(np.abs(y_test - y_pred)))
                                }
                            else:
                                y_test = st.session_state["y_test"]
                                y_pred = st.session_state["y_pred"]
                                results["metrics"] = {
                                    "accuracy": float(accuracy_score(y_test, y_pred)),
                                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                                    "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
                                }
                            
                            zip_file.writestr("results_metrics.json", json.dumps(results, indent=2, ensure_ascii=False))
                        
                        # Generate report
                        if "report" in export_options:
                            report = "=== ML VISUAL STUDIO ANALYSIS REPORT ===\n\n"
                            
                            if has_data:
                                df = st.session_state["df"]
                                report += f"RAW DATA:\n"
                                report += f"- Rows: {df.shape[0]:,}\n"
                                report += f"- Columns: {df.shape[1]}\n"
                                report += f"- Missing values: {df.isnull().sum().sum()}\n\n"
                            
                            if has_cleaned_data:
                                df_cleaned = st.session_state["df_cleaned"]
                                report += f"PREPROCESSED DATA:\n"
                                report += f"- Rows: {df_cleaned.shape[0]:,}\n"
                                report += f"- Columns: {df_cleaned.shape[1]}\n"
                                report += f"- Missing values: {df_cleaned.isnull().sum().sum()}\n\n"
                            
                            if has_model:
                                config = st.session_state["model_config"]
                                report += f"MODEL:\n"
                                report += f"- Type: {config['type']}\n"
                                report += f"- Target: {config['target']}\n"
                                report += f"- Features: {', '.join(config['features'])}\n"
                                report += f"- Learning rate: {config.get('learning_rate', 'N/A')}\n"
                                report += f"- Iterations: {config.get('n_iter', 'N/A')}\n\n"
                                
                                if "results" in export_options:
                                    report += "PERFORMANCE:\n"
                                    if config["type"] == "Regression":
                                        y_test = st.session_state["y_test"]
                                        y_pred = st.session_state["y_pred"]
                                        report += f"- MSE: {mean_squared_error(y_test, y_pred):.6f}\n"
                                        report += f"- RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.6f}\n"
                                        report += f"- R²: {r2_score(y_test, y_pred):.6f}\n"
                                    else:
                                        y_test = st.session_state["y_test"]
                                        y_pred = st.session_state["y_pred"]
                                        report += f"- Accuracy: {accuracy_score(y_test, y_pred):.4f}\n"
                                        report += f"- Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}\n"
                                        report += f"- Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}\n"
                                        report += f"- F1-Score: {f1_score(y_test, y_pred, zero_division=0):.4f}\n"
                            
                            report += f"\nReport generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            report += "Generated by ML Visual Studio\n"
                            
                            zip_file.writestr("analysis_report.txt", report)
                    
                    # Prepare download
                    zip_buffer.seek(0)
                    
                    st.success("✅ Archive generated successfully!")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download ZIP Archive",
                        data=zip_buffer.getvalue(),
                        file_name=f"ml_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                    
                    # Export summary
                    st.markdown("### 📋 Archive Contents")
                    
                    for option in export_options:
                        if option == "raw_data":
                            st.write("📊 `raw_data.csv` - Original dataset")
                        elif option == "cleaned_data":
                            st.write("🧹 `preprocessed_data.csv` - Preprocessed dataset")
                        elif option == "model":
                            st.write("🧠 `trained_model.pkl` - Serialized Python model")
                        elif option == "results":
                            st.write("📈 `results_metrics.json` - Metrics and configuration")
                        elif option == "report":
                            st.write("📄 `analysis_report.txt` - Summary report")
                
                except Exception as e:
                    st.error(f"❌ Error generating archive: {str(e)}")

# Modern footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6b7280;">
    <p style="margin: 0; font-size: 0.875rem;">
        🚀 <strong>ML Visual Studio</strong> - Artificial Intelligence Platform
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.75rem;">
        Developed with Streamlit • Version 2.0
    </p>
</div>
""", unsafe_allow_html=True)

