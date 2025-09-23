"""
Data loading utilities for breed information and styling
"""

import json
import streamlit as st


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_breed_data():
    """Load breed information from JSON with caching"""
    try:
        with open("data/breeds.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        # Comprehensive fallback data with key breeds
        return {
            "Gir": {
                "type": "indigenous", "category": "dual_purpose", "origin": "Gujarat",
                "characteristics": "Compact body, convex forehead, long pendulous ears, docile temperament",
                "milk_yield": "1200-1800 kg/lactation",
                "nutrition": {
                    "dry_matter": "2.5-3% of body weight",
                    "concentrate": "300-400g per liter of milk",
                    "green_fodder": "15-20 kg/day",
                    "water": "30-50 liters/day"
                },
                "common_diseases": ["Foot and Mouth Disease", "Mastitis", "Parasitic infections"],
                "vaccination_schedule": [
                    {"vaccine": "FMD", "frequency": "6 months", "due_in_days": 180},
                    {"vaccine": "HS", "frequency": "annual", "due_in_days": 365}
                ]
            },
            "Holstein Friesian": {
                "type": "exotic", "category": "dairy", "origin": "Netherlands",
                "characteristics": "Large size, black and white patches, very high milk producers, heat sensitive",
                "milk_yield": "7000-10000 kg/lactation",
                "nutrition": {
                    "dry_matter": "3-3.5% of body weight",
                    "concentrate": "400-500g per liter of milk",
                    "green_fodder": "25-30 kg/day",
                    "water": "80-100 liters/day"
                },
                "common_diseases": ["Mastitis", "Milk Fever", "Ketosis", "Displaced Abomasum"],
                "vaccination_schedule": [
                    {"vaccine": "FMD", "frequency": "6 months", "due_in_days": 180},
                    {"vaccine": "IBR", "frequency": "annual", "due_in_days": 365}
                ]
            },
            "Sahiwal": {
                "type": "indigenous", "category": "dairy", "origin": "Punjab/Pakistan",
                "characteristics": "Light red to brown color, drooping ears, heat tolerant, docile",
                "milk_yield": "2000-3000 kg/lactation",
                "nutrition": {
                    "concentrate": "350-450g per liter of milk",
                    "green_fodder": "20-25 kg/day",
                    "dry_matter": "2.8-3.2% of body weight",
                    "water": "40-60 liters/day"
                },
                "common_diseases": ["FMD", "Mastitis", "Tick-borne diseases"],
                "vaccination_schedule": [
                    {"vaccine": "FMD", "due_in_days": 180},
                    {"vaccine": "Anthrax", "due_in_days": 365}
                ]
            },
            "Red Sindhi": {
                "type": "indigenous", "category": "dairy", "origin": "Sindh (Pakistan)",
                "characteristics": "Red coat, compact body, heat resistant, good milk producer",
                "milk_yield": "1800-2500 kg/lactation",
                "nutrition": {
                    "concentrate": "300-400g per liter of milk",
                    "green_fodder": "18-22 kg/day"
                },
                "common_diseases": ["FMD", "Mastitis"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Tharparkar": {
                "type": "indigenous", "category": "dual_purpose", "origin": "Rajasthan",
                "characteristics": "White/light grey color, heat and drought resistant",
                "milk_yield": "1500-2200 kg/lactation",
                "nutrition": {
                    "concentrate": "250-350g per liter of milk",
                    "green_fodder": "15-20 kg/day"
                },
                "common_diseases": ["FMD", "HS"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Kankrej": {
                "type": "indigenous", "category": "dual_purpose", "origin": "Gujarat/Rajasthan",
                "characteristics": "Silver-grey color, large size, good draught capability",
                "milk_yield": "1200-1800 kg/lactation",
                "nutrition": {
                    "concentrate": "300-400g per liter of milk",
                    "green_fodder": "20-25 kg/day"
                },
                "common_diseases": ["FMD", "HS"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Jersey": {
                "type": "exotic", "category": "dairy", "origin": "Channel Islands",
                "characteristics": "Small size, fawn colored, high butterfat content",
                "milk_yield": "3500-4500 kg/lactation",
                "nutrition": {
                    "concentrate": "400-500g per liter of milk",
                    "green_fodder": "20-25 kg/day"
                },
                "common_diseases": ["Mastitis", "Milk Fever"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Crossbred (Holstein × Local)": {
                "type": "crossbred", "category": "dairy", "origin": "India",
                "characteristics": "Variable appearance, improved milk yield, moderate heat tolerance",
                "milk_yield": "3000-5000 kg/lactation",
                "nutrition": {
                    "concentrate": "350-450g per liter of milk",
                    "green_fodder": "22-28 kg/day"
                },
                "common_diseases": ["Mastitis", "FMD"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Murrah Buffalo": {
                "type": "indigenous", "category": "dairy", "origin": "Haryana",
                "characteristics": "Black color, curled horns, excellent milk producer",
                "milk_yield": "2000-3000 kg/lactation",
                "nutrition": {
                    "concentrate": "400-500g per liter of milk",
                    "green_fodder": "25-30 kg/day"
                },
                "common_diseases": ["FMD", "HS", "Mastitis"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Surti Buffalo": {
                "type": "indigenous", "category": "dairy", "origin": "Gujarat",
                "characteristics": "Light brown color, medium size, good milk quality",
                "milk_yield": "1500-2200 kg/lactation",
                "nutrition": {
                    "concentrate": "350-450g per liter of milk",
                    "green_fodder": "20-25 kg/day"
                },
                "common_diseases": ["FMD", "HS"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            }
        }


def get_breed_metadata(breed, breed_info):
    """Get comprehensive metadata for a breed"""
    data = breed_info.get(breed, {})
    
    # Format nutrition information
    nutrition_info = data.get("nutrition", {})
    nutrition_text = "**🌾 Nutrition Requirements:**\n"
    for key, value in nutrition_info.items():
        key_formatted = key.replace("_", " ").title()
        nutrition_text += f"• **{key_formatted}:** {value}\n"
    
    # Format diseases
    diseases = data.get("common_diseases", [])
    diseases_text = "**🏥 Common Diseases:**\n"
    for disease in diseases:
        diseases_text += f"• {disease}\n"
    
    # Format vaccination schedule
    vaccines = data.get("vaccination_schedule", [])
    vaccination_text = "**💉 Vaccination Schedule:**\n"
    for vaccine in vaccines:
        vaccine_name = vaccine.get("vaccine", "Unknown")
        frequency = vaccine.get("frequency", "As needed")
        vaccination_text += f"• **{vaccine_name}:** {frequency}\n"
    
    # Body weight information (placeholder)
    body_weight_text = "**⚖️ Body Weight Information:**\n"
    if data.get("type") == "indigenous":
        body_weight_text += "• **Adult Male:** 400-500 kg\n• **Adult Female:** 300-400 kg\n"
    else:
        body_weight_text += "• **Adult Male:** 600-800 kg\n• **Adult Female:** 500-700 kg\n"
    
    # Breeding information (placeholder)
    breeding_text = "**🐄 Breeding Information:**\n"
    breeding_text += "• **Age at First Calving:** 30-36 months\n"
    breeding_text += "• **Calving Interval:** 12-14 months\n"
    breeding_text += "• **Breeding Season:** Year-round\n"
    
    return {
        "origin": data.get("origin", "Unknown"),
        "category": data.get("category", "Unknown").replace("_", " ").title(),
        "type": data.get("type", "Unknown").title(),
        "characteristics": data.get("characteristics", "No information available"),
        "milk_yield": data.get("milk_yield", "Not specified"),
        "nutrition": nutrition_text,
        "diseases": diseases_text,
        "vaccination": vaccination_text,
        "body_weight": body_weight_text,
        "breeding": breeding_text
    }