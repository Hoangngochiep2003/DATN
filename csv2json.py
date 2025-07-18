import json
import os

# Đường dẫn file JSON
json_file_path = 'datafiles/fsd50k_dev_auto_caption.json'

# Bước 1: Đọc dữ liệu từ file JSON
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Bước 2: Sửa đường dẫn wav
if "data" in data:
    for item in data["data"]:
        # Lấy tên file từ đường dẫn cũ
        wav_filename = os.path.basename(item["wav"])
        # Gán đường dẫn mới
        item["wav"] = f"/content/drive/MyDrive/Datasets/FSD50K/FSD50K.dev_audio/{wav_filename}"

# Bước 3: Ghi lại vào file
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ Đã sửa đường dẫn wav sang /content/drive/MyDrive/Datasets/FSD50K/FSD50K.dev_audio/")
