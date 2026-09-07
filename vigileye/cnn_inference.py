import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class CNNDrowsinessPredictor:
    """
    MobileNetV3-Small drowsiness classifier.

    Returns probability that the detected face is Drowsy.
    """

    def __init__(self, model_path="models/drowsiness_model.pth"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {
            v: k for k, v in self.class_to_idx.items()
        }

        self.model = models.mobilenet_v3_small(weights=None)

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            2
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("[VigilEye] CNN drowsiness model loaded")
        print(f"[VigilEye] Class mapping: {self.class_to_idx}")

    def predict(self, face_bgr):
        """
        Returns:
            label: predicted class
            confidence: confidence of predicted class
            drowsy_probability: probability of Drowsy class
        """

        if face_bgr is None or face_bgr.size == 0:
            return "Unknown", 0.0, 0.0

        face_rgb = cv2.cvtColor(
            face_bgr,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(face_rgb)

        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)[0]

        prediction = torch.argmax(probabilities).item()

        label = self.idx_to_class[prediction]
        confidence = probabilities[prediction].item()

        # Find the Drowsy class regardless of class ordering
        drowsy_idx = self.class_to_idx.get("Drowsy")

        if drowsy_idx is None:
            drowsy_idx = self.class_to_idx.get("Drowsy ")

        if drowsy_idx is None:
            drowsy_probability = 0.0
        else:
            drowsy_probability = probabilities[drowsy_idx].item()

        return label, confidence, drowsy_probability