import os
from PIL import Image

# Direktori dataset
train_dir = 'datasetdeeplearning/training'
validation_dir = 'datasetdeeplearning/validation'
test_dir = 'datasetdeeplearning/testing'

# Fungsi untuk mengubah ukuran gambar dan formatnya ke JPG
def resize_and_convert_to_jpg(directory, size=(177, 177)):
    # Iterasi semua folder dalam direktori
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            # Iterasi semua file dalam subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                # Pastikan hanya memproses file gambar
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        with Image.open(file_path) as img:
                            # Konversi ke RGB jika gambar memiliki mode berbeda
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            # Ubah ukuran gambar
                            img = img.resize(size, Image.Resampling.LANCZOS)
                            # Simpan dengan format JPG (overwrite gambar asli atau ubah namanya)
                            new_file_path = os.path.splitext(file_path)[0] + ".jpg"
                            img.save(new_file_path, format="JPEG")
                            # Hapus file asli jika formatnya bukan JPG
                            if file_path != new_file_path:
                                os.remove(file_path)
                            print(f"Converted and resized: {new_file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

# Ubah format dan ukuran semua gambar dalam setiap direktori
print("Processing training images...")
resize_and_convert_to_jpg(train_dir)

print("Processing validation images...")
resize_and_convert_to_jpg(validation_dir)

print("Processing testing images...")
resize_and_convert_to_jpg(test_dir)

print("All images converted to JPG and resized to 177x177.")
