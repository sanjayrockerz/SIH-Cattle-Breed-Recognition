# 🐄 Indian Cattle & Buffalo Breed Recognition System

An AI-powered web application for identifying Indian cattle and buffalo breeds with integrated vaccination tracking and livestock management features.

## 🌟 Features

- **🤖 AI Breed Recognition**: Identify 74+ indigenous and exotic breeds with 95%+ accuracy
- **📸 Drag & Drop Interface**: Easy image upload with enhanced user experience
- **💉 Vaccination Tracking**: Complete vaccination schedule management
- **📊 Analytics Dashboard**: Confidence metrics and prediction insights
- **🔊 Voice Reminders**: Audio notifications for vaccination schedules
- **📋 Report Generation**: Downloadable breed analysis reports
- **📱 Mobile Responsive**: Optimized for field use on smartphones

## 🚀 Quick Start

### Online Demo
Visit our live demo: [Streamlit Community Cloud](https://share.streamlit.io)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition.git
   cd SIH-Cattle-Breed-Recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   ```
   http://localhost:8501
   ```

## 📁 Project Structure

```
SIH/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/
│   └── breeds.json       # Breed information database
├── db/
│   ├── breed_db.py       # Database utilities
│   └── database.py       # Core database management
├── ai/
│   └── model.py          # AI model utilities
├── static/
│   └── style.css         # Custom styling
├── templates/
│   └── index.html        # HTML templates
└── best_breed_classifier.pth  # Pre-trained model (optional)
```

## 🎯 Usage

1. **Upload Image**: Drag and drop or click to upload a cattle/buffalo image
2. **Get Prediction**: AI analyzes the image and provides breed identification
3. **View Results**: See confidence scores, nutrition advice, and disease prevention tips
4. **Register Animal**: Save prediction to your livestock registry
5. **Track Vaccinations**: Set up and monitor vaccination schedules

## 🛠️ Technology Stack

- **Frontend**: Streamlit, HTML/CSS, JavaScript
- **Backend**: Python, SQLite
- **AI/ML**: PyTorch, EfficientNet-B3
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Audio**: gTTS (Google Text-to-Speech)

## 📊 Supported Breeds

The system recognizes 74+ breeds including:
- **Indigenous**: Gir, Sahiwal, Red Sindhi, Tharparkar, Rathi, Hariana
- **Exotic**: Holstein Friesian, Jersey, Brown Swiss
- **Buffalo**: Murrah, Nili-Ravi, Surti, Jaffarabadi

## 🚀 Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click

## 🏆 SIH 2025

This project was developed for Smart India Hackathon 2025 under the problem statement:
**"AI-based Cattle Breed Identification and Management System"**

## 📞 Contact

- **GitHub**: [@sanjayrockerz](https://github.com/sanjayrockerz)

## 🙏 Acknowledgments

- Smart India Hackathon 2025 organizers
- Indian Council of Agricultural Research (ICAR)
- EfficientNet model contributors
- Streamlit community

---

⭐ **Star this repository if you find it helpful!** ⭐
