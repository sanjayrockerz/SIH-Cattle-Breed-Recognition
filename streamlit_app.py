# 🐄 SIH 2025 - Cattle & Buffalo Breed Recognition System
# Main entry point for Streamlit Community Cloud deployment

import streamlit as st
import sys
import os

# Configure Streamlit page
st.set_page_config(
    page_title="🐄 Cattle Breed Recognition - SIH 2025",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Show loading message
with st.spinner("🚀 Loading SIH 2025 Cattle Breed Recognition System..."):
    try:
        # Execute the main app.py file
        with open('app.py', 'r', encoding='utf-8') as f:
            app_code = f.read()
        
        # Execute the app code
        exec(app_code)
        
    except FileNotFoundError:
        st.error("❌ Main application file (app.py) not found!")
        st.info("Please ensure app.py is in the repository root.")
        
    except ImportError as e:
        st.error(f"❌ Import Error: {str(e)}")
        st.info("Some required packages may be missing. Trying demo mode...")
        st.code(f"Error: {e}")
        
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("There was an error loading the application.")
        st.code(f"Error details: {e}")
        
        # Show basic info as fallback
        st.markdown("## 🐄 SIH 2025 - Cattle Breed Recognition")
        st.markdown("### Emergency deployment in progress...")
        st.info("The application is being loaded. Please refresh the page in a moment.")