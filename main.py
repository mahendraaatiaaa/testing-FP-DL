from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FastAPI server...")

# Initialize FastAPI app
app = FastAPI()

# Define model architecture (SimpleNet2D)
class SimpleNet2D(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

# Initialize project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define model path
MODEL_PATH = os.path.join(project_root, "final-deep-learning", "model_training_3.pth")

# Load model instance
num_classes = 10  # Replace with your number of classes
model = SimpleNet2D(num_classes=num_classes)

# Load model state dictionary
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

# Define class labels and image dimensions
img_height = 177
img_width = 177
class_labels = ['grape', 'apple', 'starfruit', 'orange', 'kiwi', 'mango', 'pineapple', 'banana', 'watermelon', 'strawberry']

# Initialize transformation
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust with your dataset's mean and std
])

# Correct path to index.html
index_file_path = os.path.join(project_root, "final-deep-learning","index.html")

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
def read_index():
    try:
        with open(index_file_path, "r") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

# Endpoint to classify image
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty file name")

    try:
        # Read file bytes
        img_bytes = await file.read()
        
        # Open image and apply transformation
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Predict using the model
        with torch.no_grad():
            predictions = model(img_tensor)
            probabilities = torch.softmax(predictions[0], dim=0).numpy()

        # Sort results
        sorted_indices = np.argsort(probabilities)[::-1]
        results = [
            {'label': class_labels[i], 'probability': float(probabilities[i] * 100)}
            for i in sorted_indices
        ]

        return JSONResponse(content={'results': results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
static_folder = os.path.join(project_root, "final-deep-learning", "static")
app.mount("/static", StaticFiles(directory=static_folder), name="static")

# Vercel Handler
handler = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
