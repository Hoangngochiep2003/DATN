import torch
import librosa
import numpy as np

from models.clap_encoder import CLAP_Encoder
from utils import load_ss_model, parse_yaml
from pathlib import Path

def inference_single_audio(
    audio_path,
    caption,
    checkpoint_path='step=880000.ckpt',
    config_yaml='config/audiosep_base.yaml',
    device='cpu'
):
    # Load and normalize audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = audio / np.max(np.abs(audio))

    # Load config
    configs = parse_yaml(config_yaml)

    # Load query encoder (CLAP encoder)
    query_encoder = CLAP_Encoder().eval().to(device)

    # Load source separation model
    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device).eval()

    # Encode caption to embedding
    with torch.no_grad():
        condition = query_encoder.get_query_embed(
            modality='text',
            text=[caption],
            device=device
        )

        # Prepare input
        input_dict = {
            "mixture": torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(device),
            "condition": condition,
        }

        # Inference
        sep_output = pl_model.ss_model(input_dict)["waveform"]
        sep_audio = sep_output.squeeze().cpu().numpy()

    return sep_audio


# Ví dụ sử dụng
if __name__ == "__main__":
    audio_path = "mix2.wav"
    caption = "a man is speaking"
    checkpoint_path = "step=880000.ckpt"

    separated_audio = inference_single_audio(
        audio_path=audio_path,
        caption=caption,
        checkpoint_path=checkpoint_path,
        config_yaml='config/audiosep_base.yaml',
        device='cpu'
    )

    # Save kết quả
    import soundfile as sf
    sf.write("separated_output.wav", separated_audio, samplerate=16000)
