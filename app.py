# Define load_model function
from db.breed_db import breed_info
from PIL import Image
from efficientnet_pytorch import EfficientNet
import traceback
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from gtts import gTTS
from io import BytesIO
import os
import torch
import torch.nn as nn
from torchvision import transforms

import streamlit as st
from datetime import datetime, timedelta

def load_model():
    checkpoint_path = "best_breed_classifier.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    breed_classes = checkpoint.get("breed_classes")
    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(model._fc.in_features, len(breed_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, breed_classes

# Device and transform for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])


# Import breed_info from db.breed_db


# local_css function and streamlit setup moved to streamlit_app.py
def local_css(file_name="style.css"):
    try:
        with open(file_name, encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        st.sidebar.warning("Could not load style.css; using default styling.")

local_css("style.css")

def local_css(file_name="style.css"):
    try:
        with open(file_name, encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        st.sidebar.warning("Could not load style.css; using default styling.")
local_css("style.css")




# ---------------- Normalize breed_info schema ----------------
def normalize_breed_info_schema(binfo: dict) -> dict:
    """Ensure each breed entry has a consistent schema:
    keys: type, category, origin, characteristics, milk_yield,
    nutrition (dict), common_diseases (list), vaccination_schedule (list)
    """
    new = {}
    for name, meta in binfo.items():
        m = dict(meta) if isinstance(meta, dict) else {}
        m.setdefault('type', 'indigenous')
        m.setdefault('category', 'Unknown')
        m.setdefault('origin', '')
        m.setdefault('characteristics', '')
        m.setdefault('milk_yield', '')

        # Normalize nutrition into a dict for consistency
        nut = m.get('nutrition')
        if nut is None:
            m['nutrition'] = {}
        elif isinstance(nut, str):
            m['nutrition'] = {'notes': nut}
        elif isinstance(nut, dict):
            m['nutrition'] = nut
        else:
            m['nutrition'] = {'notes': str(nut)}

        # Normalize diseases into common_diseases list
        if 'common_diseases' in m and isinstance(m['common_diseases'], (list, tuple)):
            m['common_diseases'] = list(m['common_diseases'])
        elif 'disease' in m and isinstance(m['disease'], str):
            m['common_diseases'] = [m['disease']]
        else:
            m.setdefault('common_diseases', [])

        # Ensure vaccination_schedule is a list
        vs = m.get('vaccination_schedule')
        if isinstance(vs, list):
            m['vaccination_schedule'] = vs
        else:
            m.setdefault('vaccination_schedule', [])

        new[name] = m
    return new


# apply normalization so downstream code sees a consistent schema
breed_info = normalize_breed_info_schema(breed_info)


# ---------------- Prediction Function (real + demo fallback) ----------------
def predict_breed(image, model=None, breed_classes=None, demo=False, rng=None):
    if demo or model is None or breed_classes is None:
        if rng is None:
            rng = np.random.default_rng()
        if breed_classes is None:
            breed_classes = list(breed_info.keys())
        probs = rng.random(len(breed_classes))
        probs = probs / probs.sum()
        pred_idx = int(np.argmax(probs))
        breed = breed_classes[pred_idx]
        conf = float(probs[pred_idx])
        # Get formatted nutrition/disease via get_breed_meta
        _, _, nutrition_str, disease_str = get_breed_meta(breed)
        return breed, conf, nutrition_str, disease_str, probs
    # real model prediction
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    breed = breed_classes[pred_idx]
    conf = float(probs[pred_idx])
    _, _, nutrition_str, disease_str = get_breed_meta(breed)
    return breed, conf, nutrition_str, disease_str, probs

# ---------------- SQLite Setup ----------------
conn = sqlite3.connect("vaccination.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS animals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    breed TEXT,
    last_vaccination_date TEXT
)""")
conn.commit()

# ---------------- Utilities ----------------
def make_voice_and_play(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.getvalue()
    except Exception:
        return None

def make_report(breed, confidence_pct, nutrition, disease, animal_name=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    category = breed_info.get(breed, {}).get("category", "Unknown")
    lines = [
        f"Report generated: {now}",
        f"Animal: {animal_name or 'Unknown'}",
        f"Predicted breed: {breed} ({category})",
        f"Confidence: {confidence_pct:.2f}%",
        "",
        "Nutrition Advice:",
        nutrition,
        "",
        "Disease Prevention Advice:",
        disease
    ]
    return "\n".join(lines)


def _normalize_name(name: str) -> str:
    """Normalize breed names for robust matching: lower, strip, remove non-alphanum."""
    if not name:
        return ""
    import re
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


def get_breed_meta(predicted_breed: str):
    """Return a tuple (resolved_breed, category, nutrition, disease).
    Attempts exact and fuzzy matching against keys in breed_info.
    """

    if not predicted_breed:
        return None, "Unknown", "No nutrition info available.", "No disease info available."

    def _format_meta(meta: dict):
        nut = meta.get("nutrition", "")
        if isinstance(nut, dict):
            parts = []
            for k, v in nut.items():
                parts.append(f"{k.replace('_',' ').title()}: {v}")
            nutrition_str = "; ".join(parts)
        else:
            nutrition_str = str(nut) if nut else "No nutrition info available for this breed."

        dis = meta.get("disease")
        if not dis:
            cd = meta.get("common_diseases")
            if isinstance(cd, (list, tuple)):
                dis = ", ".join(cd)
        dis_str = dis if dis else "No disease info available for this breed."
        return nutrition_str, dis_str

    # exact match
    if predicted_breed in breed_info:
        meta = breed_info[predicted_breed]
        nutrition_str, dis_str = _format_meta(meta)
        return predicted_breed, meta.get("category", "Unknown"), nutrition_str, dis_str

    # normalized lookup
    norm_map = { _normalize_name(k): k for k in breed_info.keys() }
    pnorm = _normalize_name(predicted_breed)
    if pnorm in norm_map:
        real = norm_map[pnorm]
        meta = breed_info[real]
        nutrition_str, dis_str = _format_meta(meta)
        return real, meta.get("category", "Unknown"), nutrition_str, dis_str

    # substring fuzzy: if any breed_info key normalized is contained in predicted normalized
    for nk, orig in norm_map.items():
        if nk and nk in pnorm:
            meta = breed_info[orig]
            nutrition_str, dis_str = _format_meta(meta)
            return orig, meta.get("category", "Unknown"), nutrition_str, dis_str

    # reverse substring: predicted normalized in any known normalized key
    for nk, orig in norm_map.items():
        if pnorm and pnorm in nk:
            meta = breed_info[orig]
            nutrition_str, dis_str = _format_meta(meta)
            return orig, meta.get("category", "Unknown"), nutrition_str, dis_str

    # not found
    return predicted_breed, "Unknown", "No nutrition info available for this breed.", "No disease info available for this breed."


_breed_categories_cache = None
def build_breed_categories():
    """Build (and cache) mapping category -> list of breeds from breed_info."""
    global _breed_categories_cache
    if _breed_categories_cache is None:
        d = {}
        for b, meta in breed_info.items():
            cat = meta.get('category', 'Uncategorized')
            d.setdefault(cat, []).append(b)
        _breed_categories_cache = d
    return _breed_categories_cache

def get_vaccination_schedule(breed):
    # Prefer the global vaccination_schedule mapping if present
    v_sched_glob = globals().get('vaccination_schedule')
    if isinstance(v_sched_glob, dict) and breed in v_sched_glob:
        return v_sched_glob.get(breed, [])
    # Fall back to breed_info entry which may embed a vacc schedule per-breed
    meta = breed_info.get(breed, {})
    return meta.get('vaccination_schedule', [])

# ---------------- Streamlit UI Improvements ----------------

# Inject external CSS and HTML for unified UI
from pathlib import Path
css_path = Path("static/style.css")
html_path = Path("templates/index.html")
if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add custom JavaScript for enhanced drag and drop
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Find the file uploader
    const fileUploader = document.querySelector('[data-testid="stFileUploader"]');
    if (fileUploader) {
        const dropZone = fileUploader.querySelector('div');
        if (dropZone) {
            // Add drag and drop event listeners
            dropZone.addEventListener('dragover', function(e) {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('drag-over');
            });
            
            dropZone.addEventListener('dragleave', function(e) {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('drag-over');
            });
            
            dropZone.addEventListener('drop', function(e) {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('drag-over');
            });
        }
    }
});
</script>
""", unsafe_allow_html=True)

if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
                st.markdown(f.read(), unsafe_allow_html=True)

# Sidebar quick reminders (bigger)
# Sidebar HTML/CSS is now loaded from external file if needed.
today = datetime.today().date()
# compute overdue count
c.execute("SELECT name,breed,last_vaccination_date FROM animals")
animals_db = c.fetchall()
overdue_count = 0
for a in animals_db:
    try:
        last = datetime.strptime(a[2], "%Y-%m-%d").date()
    except Exception:
        continue
    for v in get_vaccination_schedule(a[1]):
        if last + timedelta(days=v.get("due_in_days", 0)) < today:
            overdue_count += 1
            break

st.sidebar.metric("Animals Registered", len(animals_db))
st.sidebar.metric("Overdue Vaccinations", overdue_count, delta=None)

# Sidebar HTML/CSS is now loaded from external file if needed.
if st.sidebar.button("➕ Register animal"):
    # replace deprecated experimental_set_query_params with st.query_params assignment
    st.query_params = {"action": ["register"]}
if st.sidebar.button("🔉 Play Reminders"):
    messages=[]
    for a in animals_db:
        name,breed,last = a
        try:
            last_date = datetime.strptime(last,"%Y-%m-%d").date()
        except Exception:
            continue
        for v in get_vaccination_schedule(breed):
            due_date = last_date + timedelta(days=v.get('due_in_days', 0))
            if due_date >= today:
                messages.append(f"{breed} {name} needs {v['vaccine']} on {due_date.strftime('%d-%b-%Y') }.")
    if messages:
        audio_bytes = make_voice_and_play(" ".join(messages), "en")
        if audio_bytes:
            st.sidebar.audio(audio_bytes, format="audio/mp3")
        else:
            st.sidebar.warning("Voice unavailable.")

# Sidebar HTML/CSS is now loaded from external file if needed.

# Main layout: left = upload/actions, right = results
left, right = st.columns([1.4, 2])

with left:
    st.markdown("### 📸 Upload Image")
    st.markdown("**Drag and drop an image or click to browse**")
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["jpg","png","jpeg"], 
        accept_multiple_files=False, 
        help="📱 Use phone camera for best results • 🐄 Center the animal in frame • 📏 JPG/PNG formats supported",
        label_visibility="collapsed"
    )
    
    # Add instructions for better results
    st.markdown("""
    <div style="background: rgba(33, 128, 141, 0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="margin: 0 0 0.5rem 0; color: var(--color-primary);">📋 Tips for Best Results:</h4>
        <ul style="margin: 0; padding-left: 1.2rem;">
            <li>🎯 Center the animal in the frame</li>
            <li>☀️ Use good lighting (natural light preferred)</li>
            <li>📐 Capture full body or clear head/face view</li>
            <li>🚫 Avoid blurry or dark images</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    use_demo = st.checkbox("🧪 Run Demo Mode (no model)", value=False, help="Enable if model file is not present or for testing.")
    analyze_btn = st.button("🔍 Analyze Image", use_container_width=True, type="primary")

    # Offer quick register modal replacement: show form when query param set or button pressed
    q = st.query_params
    if q.get("action") == ["register"]:
        with st.form("quick_register"):
            rn = st.text_input("Animal name / ID")
            rb = st.selectbox("Breed (if known)", ["--Unknown--"] + list(breed_info.keys()))
            rl = st.date_input("Last vaccination date", value=today)
            rsub = st.form_submit_button("Save")
            if rsub:
                c.execute("INSERT INTO animals (name, breed, last_vaccination_date) VALUES (?,?,?)",
                          (rn, rb if rb!="--Unknown--" else "", rl.strftime("%Y-%m-%d")))
                conn.commit()
                st.success("Registered.")

with right:
    st.markdown("### 📊 Analysis Results")
    
    # Show upload status
    if uploaded_file is not None:
        st.success(f"✅ Image uploaded: {uploaded_file.name}")
    else:
        st.info("👆 Upload an image to get started")
    
    # placeholders for dynamic update
    result_card = st.empty()
    gauge_slot = st.empty()
    top5_slot = st.empty()
    action_row = st.empty()

# analyze when button clicked or file uploaded + auto analyze
def run_analysis():
    if uploaded_file is None:
        st.warning("Please upload an image first.")
        return
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Cannot open image. Try another file.")
        return

    # show preview
    result_card.image(image, caption="📷 Preview", use_column_width=True)

    # attempt to load model
    demo_mode = use_demo
    loaded_model = None
    loaded_classes = None
    with st.spinner("Analyzing…"):
        try:
            if not demo_mode:
                loaded_model, loaded_classes = load_model()
        except FileNotFoundError:
            demo_mode = True
            st.warning("Model file missing — using demo mode.")
        except Exception:
            demo_mode = True
            st.error("Model load error — using demo mode.")
            st.text(traceback.format_exc())

        try:
            breed_pred, conf, nutrition, disease, probs = predict_breed(image, model=loaded_model, breed_classes=loaded_classes, demo=demo_mode)
        except Exception:
            st.error("Prediction failed — switching to demo.")
            breed_pred, conf, nutrition, disease, probs = predict_breed(image, demo=True)

    # Normalize and resolve breed metadata (category, nutrition, disease)
    resolved_breed, category, nutrition_resolved, disease_resolved = get_breed_meta(breed_pred)
    # use resolved values for display and storage
    breed = resolved_breed
    nutrition = nutrition_resolved
    disease = disease_resolved

    confidence_pct = conf * 100.0

    # big metric + info card
    result_html = f"""
    <div class='result-card'>
      <div class='result-grid'>
        <div class='result-left'>
          <h2>✅ {breed}</h2>
          <p><b>Confidence:</b> <span class='big-strong'>{confidence_pct:.2f}%</span></p>
          <p><b>Nutrition:</b> {nutrition}</p>
          <p><b>Disease:</b> {disease}</p>
        </div>
        <div class='result-right'>
          <button id='save-reg' class='large-btn'>💾 Save to Registry</button>
          <button id='download-report' class='large-btn secondary'>📥 Download Report</button>
        </div>
      </div>
    </div>
    """
    result_card.markdown(result_html, unsafe_allow_html=True)

    # gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        title={'text': "Confidence (%)"},
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#2E8B57"},
               'steps':[{'range':[0,50],'color':"lightcoral"},{'range':[50,80],'color':"gold"},{'range':[80,100],'color':"lightgreen"}]}
    ))
    gauge_slot.plotly_chart(gauge, use_container_width=True)

    # top5 plot
    probs = np.asarray(probs)
    if probs.ndim == 1 and probs.sum() > 0:
        top5_idx = np.argsort(probs)[-5:][::-1]
        classes = loaded_classes if loaded_classes is not None else list(breed_info.keys())
        top5_breeds = [classes[i] for i in top5_idx]
        top5_probs = [probs[i] for i in top5_idx]
        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(top5_breeds, top5_probs, color="#2E8B57")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0,1)
        ax.set_title("Top 5")
        top5_slot.pyplot(fig)
    else:
        top5_slot.info("Top-5 not available for demo output.")

    # actions: save registry and download report
    report_text = make_report(breed, confidence_pct, nutrition, disease)
    action_col1, action_col2 = action_row.columns([1,1])
    if action_col1.button("💾 Save prediction to registry"):
        with st.form("save_pred"):
            aname = st.text_input("Animal name / ID")
            # show predicted category and allow choosing category/breed
            predicted_category = breed_info.get(breed, {}).get("category", "Unknown")
            # build categories mapping on demand to avoid global name errors
            bc = build_breed_categories()
            cat_choice = st.selectbox("Category", [predicted_category] + [c for c in bc.keys() if c != predicted_category])
            # preselect predicted breed in the breed selectbox
            breed_options = [breed] + [b for b in (bc.get(cat_choice, list(breed_info.keys()))) if b != breed]
            abreed = st.selectbox("Breed detected", breed_options)
            adate = st.date_input("Last vaccination date", value=today)
            submit_save = st.form_submit_button("Save")
            if submit_save:
                c.execute("INSERT INTO animals (name, breed, last_vaccination_date) VALUES (?,?,?)",
                          (aname, abreed, adate.strftime("%Y-%m-%d")))
                conn.commit()
                st.success("Saved to registry.")
    action_col2.download_button("📥 Download report", data=report_text, file_name=f"report_{breed}{datetime.now().strftime('%Y%m%d%H%M')}.txt")

# trigger analysis
if analyze_btn:
    run_analysis()
elif uploaded_file is not None and analyze_btn == False:
    # if uploaded and user didn't press analyze, auto-run once for convenience
    run_analysis()

# Vaccination schedule display below (compact)
st.markdown("---")
st.markdown("### Upcoming vaccinations (compact)")
if animals_db:
    rows=[]
    for a in animals_db:
        name,breed,last = a
        try:
            last_date = datetime.strptime(last,"%Y-%m-%d").date()
        except Exception:
            continue
        for v in get_vaccination_schedule(breed):
            due = last_date + timedelta(days=v.get("due_in_days", 0))
            rows.append((name, breed, v["vaccine"], due.strftime("%Y-%m-%d"), "Overdue" if due<today else "Upcoming"))
    if rows:
        df = pd.DataFrame(rows, columns=["Animal","Breed","Vaccine","Due Date","Status"])
        st.dataframe(df)
    else:
        st.info("No scheduled vaccinations found.")
else:
    st.info("No registered animals yet.")