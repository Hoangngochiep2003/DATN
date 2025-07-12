import streamlit as st
import tempfile
import soundfile as sf
from inference import inference_single_audio  # Hàm bạn đã viết

# Cấu hình giao diện
st.set_page_config(page_title="🎧 AudioSep Demo", layout="centered")

# CSS tùy chỉnh
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }

    .stTextInput label {
        color: #FFD700; /* Màu chữ vàng cho label */
        font-weight: bold;
    }

    .stTextInput > div > input {
        background-color: #1e1e1e;
        color: white;
        border: 2px solid #FFD700; /* Viền vàng sáng */
        border-radius: 10px;
        padding: 0.7em;
        font-size: 1.1em;
    }

    .stTextInput > div > input:focus {
        border: 2px solid #FF6347; /* Viền đỏ khi focus */
        box-shadow: 0 0 5px #FF6347;
    }

    .stFileUploader label {
        color: #FFD700; /* Màu chữ vàng cho label */
        font-weight: bold;
    }

    .stFileUploader {
        background-color: #1e1e1e;
        border: 2px solid #FFD700; /* Viền vàng sáng */
        color: white;
        padding: 0.8em;
        border-radius: 10px;
    }

    .stFileUploader:hover {
        background-color: #333333; /* Đổi màu khi hover */
    }

    .stButton > button {
        background-color: #FFD700 !important;
        color: black !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }

    .stDownloadButton > button {
        background-color: #FFD700 !important;
        color: black !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
        margin-top: 1em;
    }

    .custom-title {
        color: #FFD700;
        font-size: 2.2em;
        font-weight: bold;
        text-align: center;
        padding: 0.5em 0;
        border-bottom: 2px solid #FFD700;
        margin-bottom: 1.5em;
    }

    </style>
""", unsafe_allow_html=True)

# Tiêu đề nổi bật
st.markdown('<div class="custom-title">🔊 AudioSep -Phân tách âm thanh theo truy vấn ngôn ngữ tự nhiên</div>', unsafe_allow_html=True)

# Upload file âm thanh
uploaded_file = st.file_uploader("📂 Tải lên file âm thanh (.wav)", type=["wav"])

# Caption nhập vào
caption = st.text_input("✍️ Mô tả nội dung cần tách", value="a man is speaking")

# Xử lý nếu có file
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.success("✅ Đã tải lên file thành công!")

    if st.button("🚀 Tách âm thanh"):
        with st.spinner("⏳ Đang xử lý..."):
            # Ghi file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # Gọi hàm tách
                output_audio = inference_single_audio(
                    audio_path=tmp_path,
                    caption=caption,
                    device='cpu'  # hoặc 'cuda'
                )

                # Lưu file đầu ra
                output_path = tmp_path.replace(".wav", "_separated.wav")
                sf.write(output_path, output_audio, samplerate=16000)

                st.success("🎉 Tách âm thanh hoàn tất!")
                st.audio(output_path, format="audio/wav")

                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Tải file kết quả", f, file_name="separated_output.wav")

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
