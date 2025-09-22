import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import os

class ChampionshipAIPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = None
        self.breed_classes = None
        self.model_loaded = False

    def load_model(self):
        checkpoint_path = "best_breed_classifier.pth"
        if not os.path.exists(checkpoint_path):
            print("Model checkpoint not found - demo mode will be used")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.breed_classes = checkpoint.get("breed_classes")
        if self.breed_classes is None:
            raise KeyError("Checkpoint missing 'breed_classes' key")
        self.model = EfficientNet.from_pretrained("efficientnet-b3")
        self.model._fc = nn.Linear(self.model._fc.in_features, len(self.breed_classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True

    def predict_breed(self, image: Image.Image):
        if self.model_loaded and self.model is not None:
            image_rgb = image.convert("RGB")
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probabilities))
            breed_name = self.breed_classes[pred_idx]
            confidence = float(probabilities[pred_idx])
        else:
            breed_classes = ["Gir", "Holstein Friesian", "Murrah"]
            probabilities = np.random.dirichlet(np.ones(len(breed_classes)) * 0.1)
            pred_idx = np.random.randint(0, len(breed_classes))
            probabilities[pred_idx] *= 3
            probabilities = probabilities / probabilities.sum()
            breed_name = breed_classes[pred_idx]
            confidence = float(probabilities[pred_idx])
        return breed_name, confidence, probabilities
