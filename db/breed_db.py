import json
import os

class BreedDatabase:
    def __init__(self, json_path=None):
        if json_path is None:
            json_path = os.path.join(os.path.dirname(__file__), '../data/breeds.json')
        with open(json_path, encoding='utf-8') as f:
            self.breed_info = json.load(f)
        self.breed_categories = self._build_category_mapping()

    def _build_category_mapping(self):
        categories = {}
        for breed_name, breed_data in self.breed_info.items():
            category = breed_data.get('category', 'Unknown')
            categories.setdefault(category, []).append(breed_name)
        return categories

    def get_breed_info(self, breed_name):
        # Exact match
        if breed_name in self.breed_info:
            return self.breed_info[breed_name]
        # Case-insensitive and fuzzy matching
        breed_lower = breed_name.lower()
        for name, info in self.breed_info.items():
            if name.lower() == breed_lower or breed_lower in name.lower():
                return info
        return None

# Load breed_info globally for easy import
try:
    _json_path = os.path.join(os.path.dirname(__file__), '../data/breeds.json')
    with open(_json_path, encoding='utf-8') as f:
        breed_info = json.load(f)
except Exception:
    # Fallback if breeds.json is missing
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
        }
    }
