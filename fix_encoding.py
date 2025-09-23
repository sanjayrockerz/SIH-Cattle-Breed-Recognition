#!/usr/bin/env python3
# Fix corrupted Unicode characters in streamlit_app.py

with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace the corrupted character
content = content.replace('st.markdown("### � Upload Cattle/Buffalo Image")', 'st.markdown("### 📷 Upload Cattle/Buffalo Image")')

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed corrupted character successfully!")