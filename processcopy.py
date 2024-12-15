import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Direktori dataset
train_dir = 'datasetdeeplearning/training'
validation_dir = 'datasetdeeplearning/validation'
test_dir = 'datasetdeeplearning/testing'

# Validasi direktori dataset
for dir_path in [train_dir, validation_dir, test_dir]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Dataset directory not found: {dir_path}")

# Hyperparameter
img_height, img_width = 177, 177
batch_size = 32

# Transformasi gambar untuk preprocessing
common_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset dan DataLoader untuk train, validasi, dan testing
train_dataset = datasets.ImageFolder(train_dir, transform=common_transform)
validation_dataset = datasets.ImageFolder(validation_dir, transform=common_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=common_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Membuat model sederhana dengan CNN
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

# Membuat model dan definisi loss function serta optimizer
num_classes = len(train_dataset.classes)
model = SimpleNet2D(num_classes=num_classes)

# Menggunakan GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Fungsi pelatihan
def train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, num_epochs=70, patience=10):
    best_val_loss = float("inf")
    best_model_weights = model.state_dict()
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds * 100

        # Validasi
        model.eval()
        val_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total_preds += labels.size(0)
                val_correct_preds += (predicted == labels).sum().item()

        val_loss /= len(validation_loader)
        val_accuracy = val_correct_preds / val_total_preds * 100

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step()

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    return model

# Melatih model
trained_model = train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, num_epochs=70)

# Evaluasi model pada data uji
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct_preds / total_preds * 100
    return test_loss, test_accuracy

# Menghitung confusion matrix dan laporan klasifikasi
def calculate_class_wise_confusion_matrix(test_loader, model):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Hitung confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    class_names = train_dataset.classes

    # Tampilkan confusion matrix untuk setiap kelas
    for idx, class_name in enumerate(class_names):
        print(f"Confusion Matrix for Class '{class_name}':")
        print(f"True Positive: {cm[idx, idx]}")
        print(f"False Positive: {sum(cm[:, idx]) - cm[idx, idx]}")
        print(f"False Negative: {sum(cm[idx, :]) - cm[idx, idx]}")
        print(f"True Negative: {sum(sum(cm)) - (sum(cm[idx, :]) + sum(cm[:, idx]) - cm[idx, idx])}")
        print("-" * 30)


    # Tampilkan laporan klasifikasi
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

# Panggil fungsi untuk menghitung confusion matrix
calculate_class_wise_confusion_matrix(test_loader, trained_model)

# Evaluasi model pada data uji
test_loss, test_accuracy = evaluate_model(trained_model, test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Menyimpan model
torch.save(trained_model.state_dict(), 'model_training_3.pth')

# Fungsi untuk mengklasifikasikan gambar
def classify_image(image_path, model):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    img_tensor = common_transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(probabilities, 1)
    class_idx = predicted.item()
    class_label = train_dataset.classes[class_idx]
    confidence = probabilities[0][class_idx].item() * 100
    print(f"Predicted Class: {class_label} with Confidence: {confidence:.2f}%")
    return class_label, confidence

# Klasifikasikan gambar
image_path = 'nanas23.jpeg'  # Ganti dengan path gambar yang sesuai
classify_image(image_path, trained_model)