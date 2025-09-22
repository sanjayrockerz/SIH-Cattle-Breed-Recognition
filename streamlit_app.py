# Streamlit App Entry Point
# This file serves as the entry point for Streamlit Community Cloud deployment

# Import and run the main application
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
from app import *

# The app.py file contains all the Streamlit code and will run automatically