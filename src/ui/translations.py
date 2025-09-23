"""
Comprehensive multilingual translation system for the cattle breed recognition app
Supports 5 Indian languages with complete UI coverage
"""

def get_translations():
    """Get the complete translation dictionary for all supported languages"""
    return {
        "English": {
            # Header and Page Title
            "page_title": "🐄 🐃 Indian Cattle & Buffalo Breed Recognition",
            "page_subtitle": "🏆 SIH 2025 - AI-Powered Livestock Management System",
            "page_description": "🤖 Advanced EfficientNet-B3 Model • 🌾 49+ Breeds • ⚡ Real-time Analysis",
            "dairy_classification": "🥛 Dairy Classification",
            "draught_identification": "🚜 Draught Identification",
            "indigenous_breeds": "🌍 Indigenous Breeds",
            "ai_model_loaded": "🤖 **AI Model**: ✅ Loaded",
            "ai_model_demo": "🤖 **AI Model**: 🔄 Demo Mode",
            
            # Dashboard
            "dashboard": "📊 Dashboard",
            "animals_registered": "🐄 Animals Registered",
            "overdue_vaccinations": "⚠️ Overdue Vaccinations",
            "register_new_animal": "➕ Register New Animal",
            
            # Upload Interface
            "upload_image": "📷 Upload Cattle/Buffalo Image",
            "drag_drop": "🖱️ Drag and drop or click to browse",
            "choose_image": "Choose an image",
            "upload_help": "📱 Use phone camera • 🐄 Center the animal • 📏 Best quality images • 🌅 Good lighting",
            "analyze_breed": "🔍 Analyze Breed",
            
            # Tips Section
            "tips_title": "💡 Tips for Best Results",
            "tip_center": "🎯 Center the animal in frame",
            "tip_lighting": "☀️ Use natural lighting",
            "tip_body": "📐 Include full body or clear face",
            "tip_avoid_blur": "🚫 Avoid blurry/dark images",
            "tip_angles": "📱 Take multiple angles if unsure",
            
            # Analysis Results
            "analysis_results": "📊 Analysis Results",
            "image_uploaded": "✅ **Image uploaded**",
            "uploaded_image": "📷 Uploaded Image",
            "analyzing_ml": "🤖 Analyzing breed with AI model...",
            "analyzing_demo": "🎲 Running demo analysis...",
            "prediction_failed": "ML prediction failed. Using demo mode.",
            
            # Validation Messages
            "validation_failed": "⚠️ **Image Validation Failed**",
            "upload_clear_image": "💡 **Please upload a clear image of cattle or buffalo**",
            "tips_valid_images": "📸 Tips for Valid Cattle/Buffalo Images",
            "good_images": "✅ Good Images:",
            "avoid_images": "❌ Avoid:",
            "clear_view": "🐄 Clear view of cattle/buffalo",
            "good_lighting": "🌅 Good lighting conditions",
            "full_body": "📐 Full or partial animal body",
            "centered": "🎯 Animal centered in frame",
            "non_animal": "🚫 Non-animal subjects",
            "too_dark": "🌙 Too dark or blurry images",
            "extreme_angles": "📐 Extreme angles",
            "low_resolution": "🔍 Very low resolution",
            
            # Prediction Results
            "predicted_breed": "🎯 Predicted Breed",
            "confidence": "📊 AI Confidence Level",
            "cattle_detected": "✅ Cattle detected",
            "not_cattle": "❌ Not a cattle/buffalo image",
            "upload_instruction": "👆 Upload an image to get started",
            "supports": "🐄 Supports cattle and buffalo breeds",
            "ai_powered": "🔬 AI-powered analysis with EfficientNet-B3",
            "mobile_optimized": "📱 Optimized for mobile photography",
            
            # Registration Form
            "register_form_title": "➕ Register New Animal",
            "animal_name": "Animal Name/ID *",
            "select_breed": "Breed *",
            "select_breed_option": "Select breed...",
            "last_vaccination_date": "Last Vaccination Date",
            "notes_optional": "Notes (optional)",
            "register_animal_btn": "💾 Register Animal",
            "registration_success": "✅ Animal registered successfully!",
            "fill_required_fields": "❌ Please fill in all required fields",
            
            # Analysis Summary
            "analysis_summary": "📋 Analysis Summary",
            "breed_identified": "Breed Identified",
            "confidence_label": "Confidence",
            "origin_label": "Origin",
            "save_to_registry": "💾 Save to Registry",
            "saved_to_registry": "✅ Saved to registry!",
            "download_report": "📄 Download Full Report"
        },
        "हिंदी (Hindi)": {
            # Header and Page Title  
            "page_title": "🐄 🐃 भारतीय गाय और भैंस की नस्ल पहचान",
            "page_subtitle": "🏆 SIH 2025 - AI-संचालित पशुधन प्रबंधन प्रणाली",
            "page_description": "🤖 उन्नत EfficientNet-B3 मॉडल • 🌾 49+ नस्लें • ⚡ तत्काल विश्लेषण",
            "dairy_classification": "🥛 डेयरी वर्गीकरण",
            "draught_identification": "🚜 खेत का काम करने वाले पशु की पहचान",
            "indigenous_breeds": "🌍 देशी नस्लें",
            "ai_model_loaded": "🤖 **AI मॉडल**: ✅ लोड हो गया",
            "ai_model_demo": "🤖 **AI मॉडल**: 🔄 डेमो मोड",
            
            # Dashboard
            "dashboard": "📊 डैशबोर्ड",
            "animals_registered": "🐄 पंजीकृत पशु",
            "overdue_vaccinations": "⚠️ बकाया टीकाकरण",
            "register_new_animal": "➕ नया पशु पंजीकृत करें",
            
            # Upload Interface
            "upload_image": "📷 गाय/भैंस की तस्वीर अपलोड करें",
            "drag_drop": "🖱️ खींचें और छोड़ें या ब्राउज़ करने के लिए क्लिक करें",
            "choose_image": "एक तस्वीर चुनें",
            "upload_help": "📱 फोन का कैमरा उपयोग करें • 🐄 पशु को केंद्र में रखें • 📏 सर्वोत्तम गुणवत्ता की तस्वीरें • 🌅 अच्छी रोशनी",
            "analyze_breed": "🔍 नस्ल का विश्लेषण करें",
            
            # Tips Section
            "tips_title": "💡 सर्वोत्तम परिणामों के लिए सुझाव",
            "tip_center": "🎯 पशु को फ्रेम के केंद्र में रखें",
            "tip_lighting": "☀️ प्राकृतिक प्रकाश का उपयोग करें",
            "tip_body": "📐 पूरा शरीर या स्पष्ट चेहरा शामिल करें",
            "tip_avoid_blur": "🚫 धुंधली/अंधेरी तस्वीरों से बचें",
            "tip_angles": "📱 यदि अनिश्चित हों तो कई कोणों से लें",
            
            # Analysis Results
            "analysis_results": "📊 विश्लेषण परिणाम",
            "image_uploaded": "✅ **तस्वीर अपलोड हुई**",
            "uploaded_image": "📷 अपलोड की गई तस्वीर",
            "analyzing_ml": "🤖 AI मॉडल से नस्ल का विश्लेषण कर रहे हैं...",
            "analyzing_demo": "🎲 डेमो विश्लेषण चला रहे हैं...",
            "prediction_failed": "ML पूर्वानुमान असफल हुआ। डेमो मोड का उपयोग कर रहे हैं।",
            
            # Validation Messages
            "validation_failed": "⚠️ **तस्वीर मान्यता असफल**",
            "upload_clear_image": "💡 **कृपया गाय या भैंस की स्पष्ट तस्वीर अपलोड करें**",
            "tips_valid_images": "📸 वैध गाय/भैंस तस्वीरों के लिए सुझाव",
            "good_images": "✅ अच्छी तस्वीरें:",
            "avoid_images": "❌ बचें:",
            "clear_view": "🐄 गाय/भैंस का स्पष्ट दृश्य",
            "good_lighting": "🌅 अच्छी प्रकाश व्यवस्था",
            "full_body": "📐 पूरा या आंशिक पशु शरीर",
            "centered": "🎯 पशु फ्रेम के केंद्र में",
            "non_animal": "🚫 गैर-पशु विषय",
            "too_dark": "🌙 बहुत अंधेरी या धुंधली तस्वीरें",
            "extreme_angles": "📐 अत्यधिक कोण",
            "low_resolution": "🔍 बहुत कम रिज़ॉल्यूशन",
            
            # Prediction Results
            "predicted_breed": "🎯 अनुमानित नस्ल",
            "confidence": "📊 AI विश्वास स्तर",
            "cattle_detected": "✅ पशु का पता चला",
            "not_cattle": "❌ यह गाय/भैंस की तस्वीर नहीं है",
            "upload_instruction": "👆 शुरू करने के लिए एक तस्वीर अपलोड करें",
            "supports": "🐄 गाय और भैंस की नस्लों का समर्थन करता है",
            "ai_powered": "🔬 EfficientNet-B3 के साथ AI-संचालित विश्लेषण",
            "mobile_optimized": "📱 मोबाइल फोटोग्राफी के लिए अनुकूलित",
            
            # Registration Form
            "register_form_title": "➕ नया पशु पंजीकृत करें",
            "animal_name": "पशु का नाम/आईडी *",
            "select_breed": "नस्ल *",
            "select_breed_option": "नस्ल चुनें...",
            "last_vaccination_date": "अंतिम टीकाकरण की तारीख",
            "notes_optional": "टिप्पणियाँ (वैकल्पिक)",
            "register_animal_btn": "💾 पशु पंजीकृत करें",
            "registration_success": "✅ पशु सफलतापूर्वक पंजीकृत हुआ!",
            "fill_required_fields": "❌ कृपया सभी आवश्यक फ़ील्ड भरें",
            
            # Analysis Summary
            "analysis_summary": "📋 विश्लेषण सारांश",
            "breed_identified": "पहचानी गई नस्ल",
            "confidence_label": "विश्वास",
            "origin_label": "मूल स्थान",
            "save_to_registry": "💾 रजिस्ट्री में सहेजें",
            "saved_to_registry": "✅ रजिस्ट्री में सहेजा गया!",
            "download_report": "📄 पूरी रिपोर्ट डाउनलोड करें"
        },
        "தமிழ் (Tamil)": {
            # Header and Page Title
            "page_title": "🐄 🐃 இந்திய மாடு மற்றும் எருமை இன அடையாளம்",
            "page_subtitle": "🏆 SIH 2025 - AI-இயங்கும் கால்நடை மேலாண்மை அமைப்பு",
            "page_description": "🤖 மேம்பட்ட EfficientNet-B3 மாதிரி • 🌾 49+ இனங்கள் • ⚡ உடனடி பகுப்பாய்வு",
            "dairy_classification": "🥛 பால் வகைப்பாடு",
            "draught_identification": "🚜 உழைப்பு மாடு அடையாளம்",
            "indigenous_breeds": "🌍 உள்நாட்டு இனங்கள்",
            "ai_model_loaded": "🤖 **AI மாதிரி**: ✅ ஏற்றப்பட்டது",
            "ai_model_demo": "🤖 **AI மாதிரி**: 🔄 டெமோ முறை",
            
            # Dashboard
            "dashboard": "📊 டாஷ்போர்டு",
            "animals_registered": "🐄 பதிவு செய்யப்பட்ட மாடுகள்",
            "overdue_vaccinations": "⚠️ தாமதமான தடுப்பூசிகள்",
            "register_new_animal": "➕ புதிய மாட்டை பதிவு செய்யவும்",
            
            # Upload Interface
            "upload_image": "📷 பசு/எருமை புகைப்படத்தை பதிவேற்றவும்",
            "drag_drop": "🖱️ இழுத்து விடுங்கள் அல்லது உலாவ கிளிக் செய்யுங்கள்",
            "choose_image": "ஒரு படத்தைத் தேர்ந்தெடுக்கவும்",
            "upload_help": "📱 போன் கேமராவைப் பயன்படுத்துங்கள் • 🐄 மாட்டை மையத்தில் வைக்கவும் • 📏 சிறந்த தரமான படங்கள் • 🌅 நல்ல வெளிச்சம்",
            "analyze_breed": "🔍 இனத்தை பகுப்பாய்வு செய்யுங்கள்",
            
            # Tips Section
            "tips_title": "💡 சிறந்த முடிவுகளுக்கான குறிப்புகள்",
            "tip_center": "🎯 மாட்டை சட்டகத்தின் மையத்தில் வைக்கவும்",
            "tip_lighting": "☀️ இயற்கை ஒளியைப் பயன்படுத்துங்கள்",
            "tip_body": "📐 முழு உடல் அல்லது தெளிவான முகத்தை சேர்க்கவும்",
            "tip_avoid_blur": "🚫 மங்கலான/இருண்ட படங்களைத் தவிர்க்கவும்",
            "tip_angles": "📱 உறுதியில்லாவிட்டால் பல கோணங்களில் எடுக்கவும்",
            
            # Analysis Results
            "analysis_results": "📊 பகுப்பாய்வு முடிவுகள்",
            "image_uploaded": "✅ **படம் பதிவேற்றப்பட்டது**",
            "uploaded_image": "📷 பதிவேற்றப்பட்ட படம்",
            "analyzing_ml": "🤖 AI மாதிரியுடன் இனத்தை பகுப்பாய்வு செய்கிறது...",
            "analyzing_demo": "🎲 டெமோ பகுப்பாய்வு இயக்குகிறது...",
            "prediction_failed": "ML முன்கணிப்பு தோல்வியுற்றது. டெமோ முறையைப் பயன்படுத்துகிறது.",
            
            # Validation Messages
            "validation_failed": "⚠️ **படம் சரிபார்ப்பு தோல்வியுற்றது**",
            "upload_clear_image": "💡 **தயவுசெய்து பசு அல்லது எருமையின் தெளிவான படத்தைப் பதிவேற்றவும்**",
            "tips_valid_images": "📸 சரியான பசு/எருமை படங்களுக்கான குறிப்புகள்",
            "good_images": "✅ நல்ல படங்கள்:",
            "avoid_images": "❌ தவிர்க்கவும்:",
            "clear_view": "🐄 பசு/எருமையின் தெளிவான காட்சி",
            "good_lighting": "🌅 நல்ல ஒளி நிலைமைகள்",
            "full_body": "📐 முழு அல்லது பகுதி மாடு உடல்",
            "centered": "🎯 மாடு சட்டகத்தின் மையத்தில்",
            "non_animal": "🚫 மாடு அல்லாத பொருள்கள்",
            "too_dark": "🌙 மிகவும் இருண்ட அல்லது மங்கலான படங்கள்",
            "extreme_angles": "📐 தீவிர கோணங்கள்",
            "low_resolution": "🔍 மிகவும் குறைந்த தெளிவுத்திறன்",
            
            # Prediction Results
            "predicted_breed": "🎯 கணிக்கப்பட்ட இனம்",
            "confidence": "📊 AI நம்பிக்கை நிலை",
            "cattle_detected": "✅ மாடு கண்டறியப்பட்டது",
            "not_cattle": "❌ இது பசு/எருமை படம் அல்ல",
            "upload_instruction": "👆 தொடங்க ஒரு படத்தைப் பதிவேற்றவும்",
            "supports": "🐄 பசு மற்றும் எருமை இனங்களை ஆதரிக்கிறது",
            "ai_powered": "🔬 EfficientNet-B3 உடன் AI-இயங்கும் பகுப்பாய்வு",
            "mobile_optimized": "📱 மொபைல் புகைப்படம் எடுப்பதற்கு உகந்தது",
            
            # Registration Form
            "register_form_title": "➕ புதிய மாட்டை பதிவு செய்யவும்",
            "animal_name": "மாட்டின் பெயர்/ஐடி *",
            "select_breed": "இனம் *",
            "select_breed_option": "இனத்தைத் தேர்ந்தெடுக்கவும்...",
            "last_vaccination_date": "கடைசி தடுப்பூசி தேதி",
            "notes_optional": "குறிப்புகள் (விருப்பம்)",
            "register_animal_btn": "💾 மாட்டைப் பதிவு செய்யவும்",
            "registration_success": "✅ மாடு வெற்றிகரமாகப் பதிவு செய்யப்பட்டது!",
            "fill_required_fields": "❌ தயவுசெய்து அனைத்து தேவையான புலங்களையும் நிரப்பவும்",
            
            # Analysis Summary
            "analysis_summary": "📋 பகுப்பாய்வு சுருக்கம்",
            "breed_identified": "அடையாளம் காணப்பட்ட இனம்",
            "confidence_label": "நம்பிக்கை",
            "origin_label": "தோற்றம்",
            "save_to_registry": "💾 பதிவகத்தில் சேமிக்கவும்",
            "saved_to_registry": "✅ பதிவகத்தில் சேமிக்கப்பட்டது!",
            "download_report": "📄 முழு அறிக்கையைப் பதிவிறக்கவும்"
        },
        "తెలుగు (Telugu)": {
            # Header and Page Title
            "page_title": "🐄 🐃 భారతీయ పశువులు మరియు గేదెల జాతుల గుర్తింపు",
            "page_subtitle": "🏆 SIH 2025 - AI-శక్తితో పశువుల నిర్వహణ వ్యవస్థ",
            "page_description": "🤖 అధునాతన EfficientNet-B3 మోడల్ • 🌾 49+ జాతులు • ⚡ తక్షణ విశ్లేషణ",
            "dairy_classification": "🥛 పాల వర్గీకరణ",
            "draught_identification": "🚜 పని చేసే పశువుల గుర్తింపు",
            "indigenous_breeds": "🌍 దేశీయ జాతులు",
            "ai_model_loaded": "🤖 **AI మోడల్**: ✅ లోడ్ అయింది",
            "ai_model_demo": "🤖 **AI మోడల్**: 🔄 డెమో మోడ్",
            
            # Dashboard
            "dashboard": "📊 డాష్‌బోర్డ్",
            "animals_registered": "🐄 నమోదైన పశువులు",
            "overdue_vaccinations": "⚠️ వాయిదా టీకాలు",
            "register_new_animal": "➕ కొత్త పశువును నమోదు చేయండి",
            
            # Upload Interface
            "upload_image": "📷 ఆవు/గేదె చిత్రాన్ని అప్‌లోడ్ చేయండి",
            "drag_drop": "🖱️ లాగి వదలండి లేదా బ్రౌజ్ చేయడానికి క్లిక్ చేయండి",
            "choose_image": "ఒక చిత్రాన్ని ఎంచుకోండి",
            "upload_help": "📱 ఫోన్ కెమెరాను ఉపయోగించండి • 🐄 పశువును మధ్యలో ఉంచండి • 📏 ఉత్తమ నాణ్యత చిత్రాలు • 🌅 మంచి వెలుతురు",
            "analyze_breed": "🔍 జాతిని విశ్లేషించండి",
            
            # Tips Section
            "tips_title": "💡 ఉత్తమ ఫలితాల కోసం చిట్కాలు",
            "tip_center": "🎯 పశువును ఫ్రేమ్ మధ్యలో ఉంచండి",
            "tip_lighting": "☀️ సహజ వెలుతురును ఉపయోగించండి",
            "tip_body": "📐 పూర్తి శరీరం లేదా స్పష్టమైన ముఖాన్ని చేర్చండి",
            "tip_avoid_blur": "🚫 అస్పష్టమైన/చీకటి చిత్రాలను నివారించండి",
            "tip_angles": "📱 అనిశ్చితంగా ఉంటే అనేక కోణాలలో తీయండి",
            
            # Analysis Results
            "analysis_results": "📊 విశ్లేషణ ఫలితాలు",
            "image_uploaded": "✅ **చిత్రం అప్‌లోడ్ అయింది**",
            "uploaded_image": "📷 అప్‌లోడ్ చేసిన చిత్రం",
            "analyzing_ml": "🤖 AI మోడల్‌తో జాతిని విశ్లేషిస్తోంది...",
            "analyzing_demo": "🎲 డెమో విశ్లేషణ నడుపుతోంది...",
            "prediction_failed": "ML అంచనా విఫలమైంది. డెమో మోడ్‌ను ఉపయోగిస్తోంది.",
            
            # Validation Messages
            "validation_failed": "⚠️ **చిత్ర ధృవీకరణ విఫలమైంది**",
            "upload_clear_image": "💡 **దయచేసి ఆవు లేదా గేదె యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి**",
            "tips_valid_images": "📸 చెల్లుబాటు అయ్యే ఆవు/గేదె చిత్రాల కోసం చిట్కాలు",
            "good_images": "✅ మంచి చిత్రాలు:",
            "avoid_images": "❌ నివారించండి:",
            "clear_view": "🐄 ఆవు/గేదె యొక్క స్పష్టమైన దృశ్యం",
            "good_lighting": "🌅 మంచి వెలుతురు పరిస్థితులు",
            "full_body": "📐 పూర్తి లేదా పాక్షిక పశు శరీరం",
            "centered": "🎯 పశువు ఫ్రేమ్ మధ్యలో",
            "non_animal": "🚫 పశువు కాని విషయాలు",
            "too_dark": "🌙 చాలా చీకటి లేదా అస్పష్టమైన చిత్రాలు",
            "extreme_angles": "📐 తీవ్రమైన కోణాలు",
            "low_resolution": "🔍 చాలా తక్కువ రిజల్యూషన్",
            
            # Prediction Results
            "predicted_breed": "🎯 అంచనా వేయబడిన జాతి",
            "confidence": "📊 AI విశ్వాస స్థాయి",
            "cattle_detected": "✅ పశువు గుర్తించబడింది",
            "not_cattle": "❌ ఇది ఆవు/గేదె చిత్రం కాదు",
            "upload_instruction": "👆 ప్రారంభించడానికి చిత్రాన్ని అప్‌లోడ్ చేయండి",
            "supports": "🐄 ఆవు మరియు గేదె జాతులకు మద్దతు ఇస్తుంది",
            "ai_powered": "🔬 EfficientNet-B3తో AI-శక్తితో విశ్లేషణ",
            "mobile_optimized": "📱 మొబైల్ ఫోటోగ్రఫీ కోసం అనుకూలీకరించబడింది",
            
            # Registration Form
            "register_form_title": "➕ కొత్త పశువును నమోదు చేయండి",
            "animal_name": "పశువు పేరు/ఐడి *",
            "select_breed": "జాతి *",
            "select_breed_option": "జాతిని ఎంచుకోండి...",
            "last_vaccination_date": "చివరి టీకా తేదీ",
            "notes_optional": "గమనికలు (ఐచ్ఛికం)",
            "register_animal_btn": "💾 పశువును నమోదు చేయండి",
            "registration_success": "✅ పశువు విజయవంతంగా నమోదు చేయబడింది!",
            "fill_required_fields": "❌ దయచేసి అన్ని అవసరమైన ఫీల్డ్‌లను నింపండి",
            
            # Analysis Summary
            "analysis_summary": "📋 విశ్లేషణ సారాంశం",
            "breed_identified": "గుర్తించబడిన జాతి",
            "confidence_label": "విశ్వాసం",
            "origin_label": "మూలం",
            "save_to_registry": "💾 రిజిస్ట్రీలో సేవ్ చేయండి",
            "saved_to_registry": "✅ రిజిస్ట్రీలో సేవ్ చేయబడింది!",
            "download_report": "📄 పూర్తి నివేదికను డౌన్‌లోడ్ చేయండి"
        },
        "ಕನ್ನಡ (Kannada)": {
            # Header and Page Title
            "page_title": "🐄 🐃 ಭಾರತೀಯ ಹಸು ಮತ್ತು ಎಮ್ಮೆ ಜಾತಿಯ ಗುರುತಿಸುವಿಕೆ",
            "page_subtitle": "🏆 SIH 2025 - AI-ಚಾಲಿತ ಪಶುಸಂಗೋಪನೆ ನಿರ್ವಹಣಾ ವ್ಯವಸ್ಥೆ",
            "page_description": "🤖 ಸುಧಾರಿತ EfficientNet-B3 ಮಾದರಿ • 🌾 49+ ಜಾತಿಗಳು • ⚡ ತ್ವರಿತ ವಿಶ್ಲೇಷಣೆ",
            "dairy_classification": "🥛 ಹಾಲು ವರ್ಗೀಕರಣ",
            "draught_identification": "🚜 ಕೆಲಸದ ಪ್ರಾಣಿಗಳ ಗುರುತಿಸುವಿಕೆ",
            "indigenous_breeds": "🌍 ಸ್ವದೇಶೀ ಜಾತಿಗಳು",
            "ai_model_loaded": "🤖 **AI ಮಾದರಿ**: ✅ ಲೋಡ್ ಆಗಿದೆ",
            "ai_model_demo": "🤖 **AI ಮಾದರಿ**: 🔄 ಡೆಮೊ ಮೋಡ್",
            
            # Dashboard
            "dashboard": "📊 ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
            "animals_registered": "🐄 ನೋಂದಾಯಿತ ಪ್ರಾಣಿಗಳು",
            "overdue_vaccinations": "⚠️ ಮುಂದೂಡಲ್ಪಟ್ಟ ಲಸಿಕೆಗಳು",
            "register_new_animal": "➕ ಹೊಸ ಪ್ರಾಣಿಯನ್ನು ನೋಂದಾಯಿಸಿ",
            
            # Upload Interface
            "upload_image": "📷 ಹಸು/ಎಮ್ಮೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
            "drag_drop": "🖱️ ಎಳೆದು ಬಿಡಿ ಅಥವಾ ಬ್ರೌಸ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ",
            "choose_image": "ಒಂದು ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ",
            "upload_help": "📱 ಫೋನ್ ಕ್ಯಾಮೆರಾವನ್ನು ಬಳಸಿ • 🐄 ಪ್ರಾಣಿಯನ್ನು ಮಧ್ಯದಲ್ಲಿ ಇರಿಸಿ • 📏 ಅತ್ಯುತ್ತಮ ಗುಣಮಟ್ಟದ ಚಿತ್ರಗಳು • 🌅 ಉತ್ತಮ ಬೆಳಕು",
            "analyze_breed": "🔍 ಜಾತಿಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
            
            # Tips Section
            "tips_title": "💡 ಅತ್ಯುತ್ತಮ ಫಲಿತಾಂಶಗಳಿಗೆ ಸಲಹೆಗಳು",
            "tip_center": "🎯 ಪ್ರಾಣಿಯನ್ನು ಚೌಕಟ್ಟಿನ ಮಧ್ಯದಲ್ಲಿ ಇರಿಸಿ",
            "tip_lighting": "☀️ ನೈಸರ್ಗಿಕ ಬೆಳಕನ್ನು ಬಳಸಿ",
            "tip_body": "📐 ಪೂರ್ಣ ದೇಹ ಅಥವಾ ಸ್ಪಷ್ಟ ಮುಖವನ್ನು ಸೇರಿಸಿ",
            "tip_avoid_blur": "🚫 ಮಂದ/ಕತ್ತಲೆಯಾದ ಚಿತ್ರಗಳನ್ನು ತಪ್ಪಿಸಿ",
            "tip_angles": "📱 ಅನುಮಾನವಿದ್ದರೆ ಅನೇಕ ಕೋನಗಳಿಂದ ತೆಗೆಯಿರಿ",
            
            # Analysis Results
            "analysis_results": "📊 ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶಗಳು",
            "image_uploaded": "✅ **ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಆಗಿದೆ**",
            "uploaded_image": "📷 ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ",
            "analyzing_ml": "🤖 AI ಮಾದರಿಯೊಂದಿಗೆ ಜಾತಿಯನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತಿದೆ...",
            "analyzing_demo": "🎲 ಡೆಮೊ ವಿಶ್ಲೇಷಣೆ ನಡೆಸುತ್ತಿದೆ...",
            "prediction_failed": "ML ಮುನ್ಸೂಚನೆ ವಿಫಲವಾಗಿದೆ. ಡೆಮೊ ಮೋಡ್ ಬಳಸುತ್ತಿದೆ.",
            
            # Validation Messages
            "validation_failed": "⚠️ **ಚಿತ್ರ ಪರಿಶೀಲನೆ ವಿಫಲವಾಗಿದೆ**",
            "upload_clear_image": "💡 **ದಯವಿಟ್ಟು ಹಸು ಅಥವಾ ಎಮ್ಮೆಯ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ**",
            "tips_valid_images": "📸 ಮಾನ್ಯವಾದ ಹಸು/ಎಮ್ಮೆ ಚಿತ್ರಗಳಿಗೆ ಸಲಹೆಗಳು",
            "good_images": "✅ ಉತ್ತಮ ಚಿತ್ರಗಳು:",
            "avoid_images": "❌ ತಪ್ಪಿಸಿ:",
            "clear_view": "🐄 ಹಸು/ಎಮ್ಮೆಯ ಸ್ಪಷ್ಟ ನೋಟ",
            "good_lighting": "🌅 ಉತ್ತಮ ಬೆಳಕಿನ ಪರಿಸ್ಥಿತಿಗಳು",
            "full_body": "📐 ಪೂರ್ಣ ಅಥವಾ ಭಾಗಶಃ ಪ್ರಾಣಿ ದೇಹ",
            "centered": "🎯 ಪ್ರಾಣಿ ಚೌಕಟ್ಟಿನ ಮಧ್ಯದಲ್ಲಿ",
            "non_animal": "🚫 ಪ್ರಾಣಿಯಲ್ಲದ ವಿಷಯಗಳು",
            "too_dark": "🌙 ತುಂಬಾ ಕಪ್ಪು ಅಥವಾ ಮಂದ ಚಿತ್ರಗಳು",
            "extreme_angles": "📐 ತೀವ್ರ ಕೋನಗಳು",
            "low_resolution": "🔍 ತುಂಬಾ ಕಡಿಮೆ ರೆಸಲ್ಯೂಶನ್",
            
            # Prediction Results
            "predicted_breed": "🎯 ಊಹಿಸಲಾದ ಜಾತಿ",
            "confidence": "📊 AI ವಿಶ್ವಾಸ ಮಟ್ಟ",
            "cattle_detected": "✅ ಪ್ರಾಣಿ ಪತ್ತೆಯಾಗಿದೆ",
            "not_cattle": "❌ ಇದು ಹಸು/ಎಮ್ಮೆ ಚಿತ್ರವಲ್ಲ",
            "upload_instruction": "👆 ಪ್ರಾರಂಭಿಸಲು ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
            "supports": "🐄 ಹಸು ಮತ್ತು ಎಮ್ಮೆ ಜಾತಿಗಳನ್ನು ಬೆಂಬಲಿಸುತ್ತದೆ",
            "ai_powered": "🔬 EfficientNet-B3 ನೊಂದಿಗೆ AI-ಚಾಲಿತ ವಿಶ್ಲೇಷಣೆ",
            "mobile_optimized": "📱 ಮೊಬೈಲ್ ಫೋಟೋಗ್ರಫಿಗಾಗಿ ಅನುಕೂಲಿತ",
            
            # Registration Form
            "register_form_title": "➕ ಹೊಸ ಪ್ರಾಣಿಯನ್ನು ನೋಂದಾಯಿಸಿ",
            "animal_name": "ಪ್ರಾಣಿಯ ಹೆಸರು/ಐಡಿ *",
            "select_breed": "ಜಾತಿ *",
            "select_breed_option": "ಜಾತಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ...",
            "last_vaccination_date": "ಕೊನೆಯ ಲಸಿಕೆ ದಿನಾಂಕ",
            "notes_optional": "ಟಿಪ್ಪಣಿಗಳು (ಐಚ್ಛಿಕ)",
            "register_animal_btn": "💾 ಪ್ರಾಣಿಯನ್ನು ನೋಂದಾಯಿಸಿ",
            "registration_success": "✅ ಪ್ರಾಣಿ ಯಶಸ್ವಿಯಾಗಿ ನೋಂದಾಯಿಸಲಾಗಿದೆ!",
            "fill_required_fields": "❌ ದಯವಿಟ್ಟು ಎಲ್ಲಾ ಅಗತ್ಯ ಕ್ಷೇತ್ರಗಳನ್ನು ಭರ್ತಿ ಮಾಡಿ",
            
            # Analysis Summary
            "analysis_summary": "📋 ವಿಶ್ಲೇಷಣೆ ಸಾರಾಂಶ",
            "breed_identified": "ಗುರುತಿಸಲಾದ ಜಾತಿ",
            "confidence_label": "ವಿಶ್ವಾಸ",
            "origin_label": "ಮೂಲ",
            "save_to_registry": "💾 ರಿಜಿಸ್ಟ್ರಿಯಲ್ಲಿ ಉಳಿಸಿ",
            "saved_to_registry": "✅ ರಿಜಿಸ್ಟ್ರಿಯಲ್ಲಿ ಉಳಿಸಲಾಗಿದೆ!",
            "download_report": "📄 ಪೂರ್ಣ ವರದಿಯನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ"
        }
    }


def get_language_translations(language):
    """Get translations for a specific language with fallback to English"""
    translations = get_translations()
    return translations.get(language, translations["English"])