import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import os


class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        datafiles=[''], 
        sampling_rate=32000, 
        max_clip_len=5,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                loaded = json.load(fp)
                if isinstance(loaded, dict) and 'data' in loaded:
                    data_json = loaded['data']
                elif isinstance(loaded, list):
                    data_json = loaded
                else:
                    raise ValueError(f"File {datafile} không đúng định dạng JSON mong muốn!")
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate

    def __len__(self):
        return len(self.all_data_json)

    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+int(self.max_length)]
        else:
            temp_wav = torch.zeros(1, int(self.max_length))
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _read_audio(self, index):
        entry = self.all_data_json[index]
        audio_path = entry.get('wav', None) if isinstance(entry, dict) else None
        if not audio_path or not os.path.exists(audio_path):
            return None, None, None
        try:
            audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)
            text = entry.get('caption', "")
            if audio_data.size(1) < self.sampling_rate * 0.5:
                raise Exception(f'{audio_path} is too short, drop it ...') 
            return text, audio_data, audio_rate
        except Exception as e:
            print(f'error: {e} occurs, when loading {audio_path}')
            return None, None, None

    def __getitem__(self, index):
        for _ in range(10):  # thử tối đa 10 lần
            text, audio_data, audio_rate = self._read_audio(index)
            if audio_data is not None and audio_rate is not None:
                break
            index = random.randint(0, len(self.all_data_json)-1)
        else:
            raise RuntimeError("Too many invalid audio entries in dataset.")
        audio_len = audio_data.shape[1] / audio_rate
        # convert stereo to single channel
        if audio_data.shape[0] > 1:
            audio_data = (audio_data[0] + audio_data[1]) / 2
        else:
            audio_data = audio_data.squeeze(0)
        # resample audio clip
        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=int(audio_rate), new_freq=self.sampling_rate)
        audio_data = audio_data.unsqueeze(0)
        audio_data = self._cut_or_randomcrop(audio_data)    # [1, N]

        # Convert waveform to mel-spectrogram for target
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate, n_fft=1024, hop_length=320, n_mels=128
        )
        mel = mel_transform(audio_data.squeeze(0))  # (n_mels, T)
        mel = mel.unsqueeze(0).transpose(1, 2)      # (1, T, n_mels)

        data_dict = {
            'text': text,
            'mixture': audio_data,   # input cho model (waveform hỗn hợp)
            'waveform': mel,         # target là mel-spectrogram
            'modality': 'audio_text'
        }
        return data_dict