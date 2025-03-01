# Here we will build frontend using streamlit.
# We will use our trained model named densenet161_v1.pth.
# We will use densenet161_v1.pth model to predict the images.

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class PlantPathogenClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = ["bacteria", "fungus", "healthy", "pests", "virus"]
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()

    def load_model(self, model_path):
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, 5)

        state_dict = torch.load(model_path, map_location=self.device)
        filtered_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        model.to(self.device)
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image = image.convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            return self.class_labels[predicted.item()]


class StreamlitApp:
    def __init__(self, model_path):
        self.classifier = PlantPathogenClassifier(model_path)

    def run(self):
        st.title("Plant Pathogen Classification")
        st.write("Upload an image of a plant leaf to predict its pathogen.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            predicted_class = self.classifier.predict(image)
            st.write(f"Predicted class: **{predicted_class}**")


if __name__ == "__main__":
    app = StreamlitApp("densenet161_v1.pth")
    app.run()
