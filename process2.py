import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the model architecture (SimpleNet2D)
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
        return x  # Pastikan ada return di akhir

# Load the trained model
num_classes = 10  # Adjust with the actual number of classes in your dataset
model = SimpleNet2D(num_classes=num_classes)
model.load_state_dict(torch.load('model_training_3.pth', map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set the model to evaluation mode

# Image preprocessing
img_height, img_width = 177, 177  # Image size should match the training size
class_labels = ['anggur', 'apel', 'belimbing', 'jeruk', 'kiwi', 'mangga', 'nanas', 'pisang', 'semangka', 'stroberi']

common_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # Resize image
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
])

def classify_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    
    # Apply transformations
    img_tensor = common_transform(img)
    
    # Add batch dimension (since PyTorch expects a batch)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():  # Turn off gradients for inference
        output = model(img_tensor)  # Forward pass

    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Sort labels by predicted probability
    sorted_indices = torch.argsort(probabilities, descending=True)
    sorted_labels = [class_labels[i] for i in sorted_indices]
    sorted_probabilities = [probabilities[i].item() for i in sorted_indices]

    # Print results
    for label, prob in zip(sorted_labels, sorted_probabilities):
        print(f'{label.upper()} {prob * 100:.2f}%')

# Example usage
image_path = 'contoh anggur.jpeg'  # Replace with the actual image path
classify_image(image_path)
