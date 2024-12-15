import torch
import torch.onnx
import onnx
import torch.nn as nn
import torch.nn.functional as F

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


# Fungsi untuk mengonversi model ke ONNX
def convert_to_onnx(model_path, img_height, img_width, onnx_filename):
    try:
        # Muat model dari file
        model = SimpleNet2D(num_classes=10)  # Inisialisasi model sesuai dengan kelas yang ada
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set model ke mode evaluasi

        # Buat tensor input dummy dengan ukuran yang sesuai
        dummy_input = torch.randn(1, 3, img_height, img_width)  # Batch size 1, 3 channel (RGB), img_height, img_width
        
        # Ekspor model ke ONNX
        torch.onnx.export(
            model,                  # Model yang akan diekspor
            dummy_input,            # Input tensor dummy
            onnx_filename,          # Nama file output ONNX
            export_params=True,     # Menyertakan parameter model dalam file ONNX
            opset_version=11,       # Versi opset ONNX (gunakan yang sesuai dengan kebutuhan Anda)
            do_constant_folding=True,  # Optimasi untuk mengurangi ukuran model
            input_names=['input'],  # Nama input pada model
            output_names=['output'],  # Nama output pada model
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Mendukung batch size dinamis
        )
        print(f"Model berhasil dikonversi ke {onnx_filename}")

        # Memverifikasi apakah model ONNX valid
        onnx.checker.check_model(onnx_filename)
        print("Model ONNX valid.")
        
    except Exception as e:
        print(f"Gagal mengonversi model: {str(e)}")


# Tentukan nama file untuk model ONNX
onnx_filename = 'model_training_4.onnx'

# Konversi model ke ONNX
if __name__ == "__main__":
    model_path = 'model_training_3.pth'  # Ganti dengan model Anda
    img_height = 177  # Ganti dengan tinggi gambar
    img_width = 177  # Ganti dengan lebar gambar
    convert_to_onnx(model_path, img_height, img_width, onnx_filename)
