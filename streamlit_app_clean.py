"""
🐄 SIH 2025 - Clean & Modular Cattle & Buffalo Breed Recognition System
Production-Ready Streamlit App with Enhanced Architecture
"""

import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import plotly.graph_objects as go

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modular components
from src.utils.data_loader import load_breed_data, get_breed_metadata
from src.utils.database import setup_database, get_animals_from_db, register_animal, get_vaccination_status
from src.models.predictor import load_ml_model, get_image_transform, predict_breed_ml, predict_breed_demo
from src.ui.translations import get_language_translations
from src.ui.styling import load_custom_css

# Configure Streamlit page (MUST be first Streamlit command)
st.set_page_config(
    page_title="🐄 Cattle Breed Recognition - SIH 2025",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition',
        'Report a bug': 'https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition/issues',
        'About': "SIH 2025 - AI-Powered Cattle Breed Recognition System"
    }
)

# Check ML availability
try:
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ ML libraries not available. Running in demo mode.")

# Initialize components
@st.cache_data
def init_app_data():
    """Initialize application data with caching"""
    breed_info = load_breed_data()
    return breed_info

# Load data and styling
breed_info = init_app_data()
css = load_custom_css()
st.markdown(css, unsafe_allow_html=True)

# Load ML model
if ML_AVAILABLE:
    model, breed_classes, device = load_ml_model()
    transform_func = get_image_transform()
    model_available = model is not None
else:
    model, breed_classes, device, transform_func = None, None, "cpu", None
    model_available = False

# Use breed_info keys as fallback classes
if breed_classes is None:
    breed_classes = sorted(list(breed_info.keys()))

# Database setup
conn, c = setup_database()

# Sidebar with language selection
st.sidebar.header("📊 Dashboard")

# Language selection at the top of sidebar
st.sidebar.markdown("### 🌐 Language / भाषा / மொழি")
language = st.sidebar.selectbox(
    "Select Language",
    options=["English", "हिंदी (Hindi)", "தமிழ் (Tamil)", "తెలుగు (Telugu)", "ಕನ್ನಡ (Kannada)"],
    index=0,
    help="Choose your preferred language"
)

# Get current language translations
t = get_language_translations(language)

# Header with enhanced farm theme - Using translations
st.markdown(f"""
<div class="hero-header">
    <h1>{t["page_title"]}</h1>
    <h3>{t["page_subtitle"]}</h3>
    <p>{t["page_description"]}</p>
    <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #1a1a1a; font-weight: 600;">
            {t["dairy_classification"]}
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #1a1a1a; font-weight: 600;">
            {t["draught_identification"]}
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #1a1a1a; font-weight: 600;">
            {t["indigenous_breeds"]}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model status indicator
model_status = t["ai_model_loaded"] if model_available else t["ai_model_demo"]
st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: rgba(32,135,147,0.1); border-radius: 8px; margin-bottom: 1rem;'>{model_status}</div>", unsafe_allow_html=True)

# Sidebar content with translations
today = datetime.today().date()
animals_db = get_animals_from_db(c)

st.sidebar.metric(t["animals_registered"], len(animals_db))

# Calculate overdue vaccinations
overdue_count = 0
for animal in animals_db:
    try:
        status, days_since = get_vaccination_status(animal[2])
        if "Overdue" in status:
            overdue_count += 1
    except:
        continue

st.sidebar.metric(t["overdue_vaccinations"], overdue_count)

if st.sidebar.button(t["register_new_animal"]):
    st.query_params = {"action": ["register"]}

# Main interface
def render_main_interface():
    """Render the main image upload and analysis interface"""
    col1, col2 = st.columns([1.3, 1.7])

    with col1:
        st.markdown(f"### {t['upload_image']}")
        st.markdown(f"**{t['drag_drop']}**")
        
        uploaded_file = st.file_uploader(
            t["choose_image"],
            type=["jpg", "jpeg", "png"],
            help=t["upload_help"],
            label_visibility="collapsed"
        )
        
        # Tips section
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, rgba(32,135,147,0.05), rgba(32,135,147,0.1)); 
                    padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #208793;">
            <h4 style="margin: 0 0 0.5rem 0; color: #208793;">{t["tips_title"]}</h4>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                <li>{t["tip_center"]}</li>
                <li>{t["tip_lighting"]}</li>
                <li>{t["tip_body"]}</li>
                <li>{t["tip_avoid_blur"]}</li>
                <li>{t["tip_angles"]}</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        
        analyze_btn = st.button(t["analyze_breed"], type="primary", use_container_width=True)

    with col2:
        st.markdown(f"### {t['analysis_results']}")
        
        if uploaded_file is not None:
            # Show upload success
            st.success(f"{t['image_uploaded']}: {uploaded_file.name}")
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption=t["uploaded_image"], use_container_width=True)
            
            if analyze_btn:
                analyze_image(image, uploaded_file.name)
        else:
            st.info(f"**{t['upload_instruction']}**")
            st.markdown(f'''
            <div style="text-align: center; padding: 2rem; color: #666;">
                <p>{t["supports"]}</p>
                <p>{t["ai_powered"]}</p>
                <p>{t["mobile_optimized"]}</p>
            </div>
            ''', unsafe_allow_html=True)

def analyze_image(image, filename):
    """Analyze the uploaded image for breed prediction"""
    with st.spinner(t["analyzing_ml"] if model_available else t["analyzing_demo"]):
        # Prediction with cattle validation
        if model_available:
            breed, conf, probs, validation_message = predict_breed_ml(
                image, model, breed_classes, device, transform_func
            )
            if breed is None:
                breed, conf, probs, validation_message = predict_breed_demo(image, breed_classes)
                if breed is None:
                    handle_validation_failure(validation_message)
                    return
                else:
                    st.warning(t["prediction_failed"])
        else:
            breed, conf, probs, validation_message = predict_breed_demo(image, breed_classes)
            if breed is None:
                handle_validation_failure(validation_message)
                return
        
        # Display validation status
        st.info(validation_message)
        
        # Display results
        display_prediction_results(breed, conf, probs, filename, image)

def handle_validation_failure(validation_message):
    """Handle cases where image validation fails"""
    st.error(t["validation_failed"])
    st.error(validation_message)
    st.info(t["upload_clear_image"])
    
    # Enhanced guidance with visual styling
    st.markdown(f'''
    <div style="background: linear-gradient(135deg, #FFC107 0%, #4CAF50 100%); 
                padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white;">
        <h4 style="margin: 0 0 1rem 0;">{t["tips_valid_images"]}</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <p><strong>{t["good_images"]}</strong></p>
                <ul>
                    <li>{t["clear_view"]}</li>
                    <li>{t["good_lighting"]}</li>
                    <li>{t["full_body"]}</li>
                    <li>{t["centered"]}</li>
                </ul>
            </div>
            <div>
                <p><strong>{t["avoid_images"]}</strong></p>
                <ul>
                    <li>{t["non_animal"]}</li>
                    <li>{t["too_dark"]}</li>
                    <li>{t["extreme_angles"]}</li>
                    <li>{t["low_resolution"]}</li>
                </ul>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def display_prediction_results(breed, conf, probs, filename, image):
    """Display comprehensive prediction results"""
    confidence_pct = conf * 100
    
    # Basic results
    st.success(f"🎯 **{t['predicted_breed']}:** {breed}")
    st.info(f"📊 **{t['confidence']}:** {confidence_pct:.1f}%")
    
    # Get breed metadata
    metadata = get_breed_metadata(breed, breed_info)
    
    # Basic information cards
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.metric("🌍 Geographic Origin", metadata['origin'])
        st.metric("🏷️ Breed Category", metadata['category'])
    
    with col_info2:
        st.metric("🐄 Animal Type", metadata['type'])
        st.metric("🥛 Average Milk Yield", metadata['milk_yield'])
    
    # Characteristics section
    st.markdown("### 🔍 Physical Characteristics")
    st.markdown(f'''
    <div style="background: linear-gradient(135deg, #8D6E63 0%, #42A5F5 100%); 
                padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;">
        {metadata["characteristics"]}
    </div>
    ''', unsafe_allow_html=True)
    
    # Confidence gauge
    display_confidence_gauge(confidence_pct)
    
    # Detailed information tabs
    display_detail_tabs(metadata, probs, breed_classes)
    
    # Action buttons
    display_action_buttons(breed, confidence_pct, metadata, filename)

def display_confidence_gauge(confidence_pct):
    """Display confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        title={'text': f"{t['confidence']} (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#208793"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 80], 'color': "gold"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_detail_tabs(metadata, probs, breed_classes):
    """Display detailed information in tabs"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🥗 Nutrition", "🏥 Health", "💉 Vaccination", "🐄 Breeding", "📊 Analysis"])
    
    with tab1:
        st.subheader("🌾 Nutrition Requirements")
        st.markdown(metadata.get('nutrition', 'Complete nutrition data will be shown here.'))
    
    with tab2:
        st.subheader("🏥 Health & Disease Management")
        st.markdown(metadata.get('diseases', 'Disease information will be shown here.'))
    
    with tab3:
        st.subheader("💉 Vaccination Schedule")
        st.markdown(metadata.get('vaccination', 'Vaccination schedule will be shown here.'))
    
    with tab4:
        st.subheader("🐄 Breeding Information")
        st.markdown(metadata.get('breeding', 'Breeding information will be shown here.'))
    
    with tab5:
        st.markdown("### 📊 Prediction Analysis")
        if len(probs) > 1:
            # Top 5 predictions
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5_breeds = [breed_classes[i] for i in top5_idx]
            top5_probs = [probs[i] * 100 for i in top5_idx]
            
            chart_data = pd.DataFrame({
                'Breed': top5_breeds,
                'Confidence (%)': top5_probs
            })
            
            st.bar_chart(chart_data.set_index('Breed'), height=300)
            
            # Show detailed comparison
            st.markdown("#### 🔍 Top Predictions Comparison")
            for i, (breed_name, prob) in enumerate(zip(top5_breeds, top5_probs)):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                st.markdown(f"{emoji} **{breed_name}**: {prob:.1f}%")
        else:
            st.info("Analysis data not available for detailed comparison.")

def display_action_buttons(breed, confidence_pct, metadata, filename):
    """Display action buttons for save and download"""
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button(t["save_to_registry"], use_container_width=True):
            success = register_animal(
                c, conn, 
                f"Animal_{len(get_animals_from_db(c))+1}", 
                breed, 
                today.strftime("%Y-%m-%d"), 
                f"Confidence: {confidence_pct:.1f}%"
            )
            if success:
                st.success(t["saved_to_registry"])
    
    with col_b:
        # Generate comprehensive report
        report_content = generate_report(breed, confidence_pct, metadata, filename)
        
        # Enhanced visual summary
        display_summary_cards(breed, confidence_pct, metadata)
        
        st.download_button(
            t["download_report"],
            data=report_content,
            file_name=f"breed_report_{breed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def generate_report(breed, confidence_pct, metadata, filename):
    """Generate comprehensive analysis report"""
    return f"""SIH 2025 - CATTLE BREED ANALYSIS REPORT

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image File: {filename}
Generated by: AI-Powered Cattle Breed Recognition System

PREDICTION RESULTS
Predicted Breed: {breed}
Confidence Level: {confidence_pct:.2f}%
AI Model: EfficientNet-B3 Deep Learning Model

BREED INFORMATION
Origin: {metadata['origin']}
Category: {metadata['category']}
Type: {metadata['type']}
Milk Yield: {metadata['milk_yield']}

Physical Characteristics:
{metadata['characteristics']}

NUTRITION MANAGEMENT
{metadata.get('nutrition', 'Nutrition data not available').replace('**', '').replace('🌾', '')}

HEALTH & DISEASE MANAGEMENT
{metadata.get('diseases', 'Disease data not available').replace('**', '').replace('🏥', '')}

VACCINATION SCHEDULE
{metadata.get('vaccination', 'Vaccination data not available').replace('**', '').replace('💉', '')}

BREEDING INFORMATION
{metadata.get('breeding', 'Breeding data not available').replace('**', '').replace('🐄', '')}

RECOMMENDATIONS
1. Follow the nutrition guidelines strictly for optimal milk production
2. Maintain regular vaccination schedule as per the recommended timeline
3. Monitor for common diseases and consult veterinarian for preventive care
4. Ensure adequate water supply and quality fodder throughout the year
5. Maintain proper breeding records for genetic improvement

DISCLAIMER
This analysis is generated by an AI system for educational and advisory purposes. 
Always consult with qualified veterinarians and livestock experts for medical decisions and breeding programs.

CONTACT INFORMATION
Project: Smart India Hackathon 2025
Team: Nexel
GitHub: https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition
Email: myteamcreations09@gmail.com

Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}"""

def display_summary_cards(breed, confidence_pct, metadata):
    """Display visual summary cards"""
    st.markdown(f"### {t['analysis_summary']}")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #FFC107 100%); 
                    padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0;">🎯</h3>
            <p style="margin: 0;"><strong>{t["breed_identified"]}</strong></p>
            <p style="margin: 0; font-size: 0.9rem;">{breed}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #42A5F5 0%, #4CAF50 100%); 
                    padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0;">📊</h3>
            <p style="margin: 0;"><strong>{t["confidence_label"]}</strong></p>
            <p style="margin: 0; font-size: 0.9rem;">{confidence_pct:.1f}%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with summary_col3:
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #8D6E63 0%, #42A5F5 100%); 
                    padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0;">🌍</h3>
            <p style="margin: 0;"><strong>{t["origin_label"]}</strong></p>
            <p style="margin: 0; font-size: 0.9rem;">{metadata["origin"]}</p>
        </div>
        ''', unsafe_allow_html=True)

def render_registration_form():
    """Render the animal registration form"""
    st.markdown("---")
    with st.form("register_animal", clear_on_submit=True):
        st.markdown(f"### {t['register_form_title']}")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            name = st.text_input(t["animal_name"])
            breed = st.selectbox(t["select_breed"], [t["select_breed_option"]] + sorted(breed_classes))
        
        with col_r2:
            last_vacc = st.date_input(t["last_vaccination_date"], value=today)
            notes = st.text_area(t["notes_optional"], height=100)
        
        submitted = st.form_submit_button(t["register_animal_btn"], type="primary")
        
        if submitted:
            if name and breed != t["select_breed_option"]:
                success = register_animal(c, conn, name, breed, last_vacc.strftime("%Y-%m-%d"), notes)
                if success:
                    st.success(f"✅ **{name}** {t['registration_success']}")
                    st.query_params.clear()
            else:
                st.error(f"{t['fill_required_fields']} (*)")

def render_animals_registry():
    """Render the animals registry view"""
    animals_db = get_animals_from_db(c)
    if len(animals_db) > 0:
        st.markdown("---")
        st.markdown("### 📋 Registered Animals")
        
        # Convert to DataFrame for better display
        df_animals = pd.DataFrame(animals_db, columns=["Name", "Breed", "Last Vaccination"])
        df_animals["Days Since Vaccination"] = df_animals["Last Vaccination"].apply(
            lambda x: get_vaccination_status(x)[1]
        )
        df_animals["Status"] = df_animals["Last Vaccination"].apply(
            lambda x: get_vaccination_status(x)[0]
        )
        
        st.dataframe(df_animals, use_container_width=True)

def render_footer():
    """Render the application footer"""
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; padding: 2rem; 
                background: linear-gradient(135deg, rgba(32,135,147,0.05), rgba(32,135,147,0.1)); 
                border-radius: 12px; margin-top: 2rem;">
        <h3 style="color: #208793; margin-bottom: 1rem;">🏆 Smart India Hackathon 2025</h3>
        <p style="margin: 0.5rem 0;"><strong>AI-based Cattle Breed Identification and Management System</strong></p>
        <p style="margin: 0.5rem 0;">Developed by <strong>Team Nexel</strong></p>
        <p style="margin: 0.5rem 0;">
            <a href="https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition" target="_blank" 
               style="color: #208793; text-decoration: none;">🔗 GitHub Repository</a> • 
            <a href="mailto:myteamcreations09@gmail.com" style="color: #208793; text-decoration: none;">✉️ Contact</a>
        </p>
        <p style="font-size: 0.9rem; color: #666; margin-top: 1rem;">
            Empowering farmers with AI • Supporting indigenous breeds • Building the future of livestock management
        </p>
    </div>
    ''', unsafe_allow_html=True)

# Main application flow
def main():
    """Main application entry point"""
    # Main interface
    render_main_interface()
    
    # Registration form (if requested)
    q = st.query_params
    if q.get("action") == ["register"]:
        render_registration_form()
    
    # Animals registry view
    render_animals_registry()
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()