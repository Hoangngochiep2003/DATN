import json

# Đường dẫn file JSON
json_file_path = 'datafiles/fsd50k_dev_auto_caption.json'

# Bước 1: Đọc dữ liệu từ file JSON
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Bước 2: Xử lý đường dẫn wav
if "data" in data:
    for item in data["data"]:
        item["wav"] = item["wav"].replace("fsd/", "")

# Bước 3: Ghi lại vào file
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ Đã bỏ 'fsd/' khỏi tất cả đường dẫn wav.")
