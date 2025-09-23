"""
Enhanced farm-inspired CSS styling with mobile optimization
"""

import streamlit as st


@st.cache_data
def load_custom_css():
    """Enhanced farm-inspired CSS with vibrant colors and accessibility"""
    
    return """
    <style>
    /* SIH 2025 - Enhanced Farm-Inspired Color Palette */
    :root {
        --farm-green-primary: #2E7D32;
        --farm-green-light: #4CAF50;
        --farm-green-dark: #1B5E20;
        --farm-yellow-primary: #F57F17;
        --farm-yellow-light: #FFC107;
        --farm-yellow-dark: #E65100;
        --farm-brown-primary: #5D4037;
        --farm-brown-light: #8D6E63;
        --farm-brown-dark: #3E2723;
        --farm-blue-primary: #1976D2;
        --farm-blue-light: #42A5F5;
        --farm-blue-dark: #0D47A1;
        --farm-cream: #FFF8E1;
        --farm-earth: #F3E5AB;
        --text-dark: #2E2E2E;
        --text-light: #FFFFFF;
        --success-green: #4CAF50;
        --warning-orange: #FF9800;
        --error-red: #F44336;
        --shadow-soft: 0 4px 12px rgba(0,0,0,0.1);
        --shadow-strong: 0 8px 24px rgba(0,0,0,0.15);
        --border-radius: 12px;
        --border-radius-large: 20px;
    }

    /* Performance optimizations */
    * { box-sizing: border-box; }
    
    /* Global app styling */
    .stApp {
        background: linear-gradient(135deg, var(--farm-cream) 0%, var(--farm-earth) 100%);
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 1rem !important;
        max-width: 1200px !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .stApp > header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }

    /* Enhanced header with farm theme */
    .hero-header {
        background: linear-gradient(135deg, var(--farm-green-primary) 0%, var(--farm-blue-primary) 100%);
        color: var(--text-light);
        padding: 2rem;
        border-radius: var(--border-radius-large);
        text-align: center;
        box-shadow: var(--shadow-strong);
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }

    .hero-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .hero-header h3 {
        margin: 0.5rem 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }

    .hero-header p {
        margin: 0.5rem 0;
        font-size: 1rem;
        opacity: 0.8;
    }

    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--farm-green-light) 0%, var(--farm-green-primary) 100%) !important;
        border-radius: 0 var(--border-radius-large) var(--border-radius-large) 0 !important;
        padding: 1.5rem 1rem !important;
    }

    /* Farm-themed metrics */
    .stMetric {
        background: var(--text-light) !important;
        padding: 1rem !important;
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-soft) !important;
        border-left: 4px solid var(--farm-green-primary) !important;
        margin-bottom: 1rem !important;
        transition: transform 0.2s ease !important;
    }

    .stMetric:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-strong) !important;
    }

    .stMetric label {
        color: var(--farm-green-dark) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .stMetric > div > div {
        color: var(--farm-brown-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Enhanced file uploader */
    .stFileUploader > div {
        background: linear-gradient(135deg, var(--text-light) 0%, var(--farm-cream) 100%) !important;
        border: 3px dashed var(--farm-green-primary) !important;
        border-radius: var(--border-radius-large) !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stFileUploader > div:hover {
        border-color: var(--farm-blue-primary) !important;
        background: linear-gradient(135deg, var(--farm-cream) 0%, var(--farm-earth) 100%) !important;
        transform: scale(1.02) !important;
    }

    .stFileUploader > div > div {
        border: none !important;
        background: transparent !important;
    }

    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--farm-green-primary) 0%, var(--farm-green-light) 100%) !important;
        color: var(--text-light) !important;
        border: none !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-soft) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--farm-blue-primary) 0%, var(--farm-blue-light) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-strong) !important;
    }

    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.5rem !important;
        gap: 0.5rem !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: var(--border-radius) !important;
        color: var(--text-dark) !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--farm-green-primary) 0%, var(--farm-blue-primary) 100%) !important;
        color: var(--text-light) !important;
        box-shadow: var(--shadow-soft) !important;
    }

    /* Enhanced success/error messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--success-green) 0%, var(--farm-green-light) 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
        animation: slideInRight 0.5s ease !important;
    }

    .stError {
        background: linear-gradient(135deg, var(--error-red) 0%, #E57373 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
        animation: shake 0.5s ease !important;
    }

    .stInfo {
        background: linear-gradient(135deg, var(--farm-blue-primary) 0%, var(--farm-blue-light) 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .stWarning {
        background: linear-gradient(135deg, var(--warning-orange) 0%, var(--farm-yellow-primary) 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
    }

    /* Progress indicators */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--farm-green-primary) 0%, var(--farm-yellow-primary) 100%) !important;
        border-radius: var(--border-radius) !important;
    }

    /* Form styling */
    .stForm {
        border: 1px solid var(--farm-green-primary) !important;
        border-radius: var(--border-radius) !important;
        background: var(--text-light) !important;
        padding: 1.5rem !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        border-radius: var(--border-radius) !important;
        border: 2px solid var(--farm-green-light) !important;
        transition: border-color 0.3s ease !important;
    }

    .stTextInput > div > div:focus, .stTextArea > div > div:focus {
        border-color: var(--farm-blue-primary) !important;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2) !important;
    }

    /* Image styling */
    .stImage {
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-soft) !important;
        overflow: hidden !important;
    }

    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        50% { transform: translateX(5px); }
        75% { transform: translateX(-5px); }
        100% { transform: translateX(0); }
    }

    /* Mobile responsiveness with enhanced text visibility */
    @media (max-width: 768px) {
        /* Enhanced text contrast for mobile */
        .stApp {
            background: linear-gradient(135deg, #F5F5F5 0%, #E8E8E8 100%) !important;
        }
        
        /* Dark, bold text for better mobile visibility */
        h1, h2, h3, h4, h5, h6 {
            color: #1A1A1A !important;
            font-weight: 800 !important;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.8) !important;
        }
        
        /* Enhanced paragraph and body text */
        p, div, span, label {
            color: #2C2C2C !important;
            font-weight: 600 !important;
            text-shadow: 0.5px 0.5px 1px rgba(255,255,255,0.8) !important;
        }
        
        /* Stronger contrast for metrics and important text */
        .stMetric label {
            color: #0D4A0D !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.9) !important;
        }
        
        .stMetric > div > div {
            color: #1A1A1A !important;
            font-size: 2rem !important;
            font-weight: 900 !important;
            text-shadow: 1px 1px 3px rgba(255,255,255,0.8) !important;
        }
        
        /* Enhanced button text visibility */
        .stButton > button {
            background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
            border: 2px solid #1B5E20 !important;
            padding: 0.5rem 1rem !important;
            font-size: 1rem !important;
        }
        
        /* Enhanced tab text */
        .stTabs [data-baseweb="tab"] {
            color: #1A1A1A !important;
            font-weight: 800 !important;
            text-shadow: 0.5px 0.5px 1px rgba(255,255,255,0.8) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
        }
        
        /* Enhanced form text */
        .stTextInput label, .stSelectbox label, .stTextArea label {
            color: #1A1A1A !important;
            font-weight: 700 !important;
            text-shadow: 0.5px 0.5px 1px rgba(255,255,255,0.8) !important;
        }
        
        /* Enhanced success/error message text */
        .stSuccess, .stError, .stInfo, .stWarning {
            font-weight: 700 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.6) !important;
        }
        
        /* Enhanced sidebar text for mobile */
        .css-1d391kg * {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
        }
        
        /* Enhanced markdown text */
        .stMarkdown {
            color: #1A1A1A !important;
            font-weight: 600 !important;
        }
        
        /* Enhanced file uploader text */
        .stFileUploader label {
            color: #1A1A1A !important;
            font-weight: 700 !important;
            text-shadow: 0.5px 0.5px 1px rgba(255,255,255,0.8) !important;
        }
        
        .hero-header h1 {
            font-size: 2rem !important;
            color: #FFFFFF !important;
            font-weight: 900 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.9) !important;
        }
        
        .hero-header h3, .hero-header p {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8) !important;
        }
        
        .stFileUploader > div {
            padding: 1.5rem !important;
            background: #FFFFFF !important;
            border: 3px solid #2E7D32 !important;
        }
        
        .stMetric {
            margin-bottom: 0.5rem !important;
            background: #FFFFFF !important;
            border: 2px solid #2E7D32 !important;
        }
        
        .main .block-container {
            padding: 1rem 0.5rem !important;
        }
    }

    /* Accessibility improvements */
    .stButton > button:focus,
    .stTabs [data-baseweb="tab"]:focus {
        outline: 3px solid var(--farm-yellow-primary) !important;
        outline-offset: 2px !important;
    }

    /* High contrast mode */
    @media (prefers-contrast: high) {
        :root {
            --farm-green-primary: #1B5E20;
            --farm-blue-primary: #0D47A1;
            --text-dark: #000000;
        }
    }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """