import os
import requests

# 1. Tạo thư mục nếu chưa có
os.makedirs("checkpoint", exist_ok=True)

# 2. Sửa dòng load_state_dict trong factory.py
factory_path = os.path.join("models", "CLAP", "open_clip", "factory.py")
with open(factory_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(factory_path, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line.replace("model.load_state_dict(ckpt)", "model.load_state_dict(ckpt, strict=False)"))

# 3. Tải checkpoint về
url = "https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt"
dest_path = os.path.join("checkpoint", "music_speech_audioset_epoch_15_esc_89.98.pt")

print("⏳ Downloading checkpoint...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("✅ Done.")
