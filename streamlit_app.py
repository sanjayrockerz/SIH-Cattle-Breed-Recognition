# 🐄 SIH 2025 - Cattle & Buffalo Breed Recognition System
# Streamlit Community Cloud Deployment

import streamlit as st
import os
import sys
import json
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(
    page_title="🐄 Cattle Breed Recognition - SIH 2025",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
try:
    with open("static/style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

# Load breed data
try:
    with open("data/breeds.json", "r", encoding="utf-8") as f:
        breed_info = json.load(f)
except:
    # Fallback breed data
    breed_info = {
        "Gir": {
            "type": "indigenous", 
            "category": "dual_purpose", 
            "origin": "Gujarat",
            "characteristics": "Compact body, convex forehead, long pendulous ears",
            "milk_yield": "1200-1800 kg/lactation",
            "nutrition": {"concentrate": "300-400g per liter of milk"},
            "common_diseases": ["Foot and Mouth Disease", "Mastitis"],
            "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
        },
        "Holstein Friesian": {
            "type": "exotic",
            "category": "dairy", 
            "origin": "Netherlands",
            "characteristics": "Large size, black and white patches",
            "milk_yield": "7000-10000 kg/lactation",
            "nutrition": {"concentrate": "400-500g per liter of milk"},
            "common_diseases": ["Mastitis", "Milk Fever"],
            "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
        }
    }

# Demo prediction function (no model required)
def predict_breed_demo(image):
    """Demo prediction function that works without the ML model"""
    breeds = list(breed_info.keys())
    np.random.seed(42)  # For consistent demo results
    probs = np.random.random(len(breeds))
    probs = probs / probs.sum()
    pred_idx = int(np.argmax(probs))
    breed = breeds[pred_idx]
    conf = float(probs[pred_idx])
    
    # Get breed info
    meta = breed_info.get(breed, {})
    nutrition = meta.get("nutrition", {})
    if isinstance(nutrition, dict):
        nutrition_str = "; ".join([f"{k}: {v}" for k, v in nutrition.items()])
    else:
        nutrition_str = str(nutrition)
    
    diseases = meta.get("common_diseases", [])
    disease_str = ", ".join(diseases) if diseases else "No specific diseases listed"
    
    return breed, conf, nutrition_str, disease_str, probs

# SQLite setup
conn = sqlite3.connect("vaccination.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS animals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    breed TEXT,
    last_vaccination_date TEXT
)""")
conn.commit()

# Main App Interface
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🐄 Indian Cattle & Buffalo Breed Recognition</h1>
    <p style="font-size: 1.2rem; color: #666;">
        SIH 2025 - AI-Powered Livestock Management System
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar stats
today = datetime.today().date()
c.execute("SELECT name,breed,last_vaccination_date FROM animals")
animals_db = c.fetchall()

st.sidebar.metric("🐄 Animals Registered", len(animals_db))
st.sidebar.metric("💉 Vaccination Reminders", "Demo Mode")

if st.sidebar.button("➕ Register New Animal"):
    st.query_params = {"action": ["register"]}

# Main layout
left, right = st.columns([1.4, 2])

with left:
    st.markdown("### 📸 Upload Image")
    st.markdown("**Drag and drop an image or click to browse**")
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["jpg","png","jpeg"], 
        accept_multiple_files=False, 
        help="📱 Use phone camera • 🐄 Center the animal • 📏 JPG/PNG supported",
        label_visibility="collapsed"
    )
    
    # Tips
    st.markdown("""
    <div style="background: rgba(33, 128, 141, 0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="margin: 0 0 0.5rem 0; color: #208793;">📋 Tips for Best Results:</h4>
        <ul style="margin: 0; padding-left: 1.2rem;">
            <li>🎯 Center the animal in the frame</li>
            <li>☀️ Use good lighting</li>
            <li>📐 Capture full body or clear head view</li>
            <li>🚫 Avoid blurry images</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    analyze_btn = st.button("🔍 Analyze Image", use_container_width=True, type="primary")

with right:
    st.markdown("### 📊 Analysis Results")
    
    if uploaded_file is not None:
        st.success(f"✅ Image uploaded: {uploaded_file.name}")
        
        # Show image preview
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🤖 Analyzing breed..."):
                # Demo prediction
                breed, conf, nutrition, disease, probs = predict_breed_demo(image)
                confidence_pct = conf * 100.0
                
                # Results
                st.success(f"🎯 **Predicted Breed:** {breed}")
                st.info(f"🎯 **Confidence:** {confidence_pct:.1f}%")
                
                # Details in expandable sections
                with st.expander("🥗 Nutrition Information"):
                    st.write(nutrition)
                
                with st.expander("🏥 Health & Disease Prevention"):
                    st.write(disease)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence_pct,
                    title={'text': "Confidence (%)"},
                    gauge={'axis':{'range':[0,100]}, 
                           'bar':{'color':"#208793"},
                           'steps':[{'range':[0,50],'color':"lightcoral"},
                                   {'range':[50,80],'color':"gold"},
                                   {'range':[80,100],'color':"lightgreen"}]}
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save to Registry"):
                        # Simple save (demo)
                        c.execute("INSERT INTO animals (name, breed, last_vaccination_date) VALUES (?,?,?)",
                                 (f"Animal_{len(animals_db)+1}", breed, today.strftime("%Y-%m-%d")))
                        conn.commit()
                        st.success("Saved!")
                
                with col2:
                    report = f"""
SIH 2025 - Breed Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Predicted Breed: {breed}
Confidence: {confidence_pct:.1f}%
Nutrition: {nutrition}
Disease Prevention: {disease}
                    """
                    st.download_button("📥 Download Report", 
                                     data=report, 
                                     file_name=f"breed_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    else:
        st.info("👆 Upload an image to get started")

# Registration form
q = st.query_params
if q.get("action") == ["register"]:
    with st.form("register_animal"):
        st.markdown("### ➕ Register New Animal")
        name = st.text_input("Animal Name/ID")
        breed = st.selectbox("Breed", ["Select..."] + list(breed_info.keys()))
        last_vacc = st.date_input("Last Vaccination Date", value=today)
        
        if st.form_submit_button("Save Animal"):
            if name and breed != "Select...":
                c.execute("INSERT INTO animals (name, breed, last_vaccination_date) VALUES (?,?,?)",
                         (name, breed, last_vacc.strftime("%Y-%m-%d")))
                conn.commit()
                st.success(f"✅ {name} registered successfully!")
                st.query_params.clear()
            else:
                st.error("Please fill all fields")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🏆 <strong>Smart India Hackathon 2025</strong> - AI-based Cattle Breed Identification</p>
    <p>Developed by Team SanjayRockerz | 
    <a href="https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)