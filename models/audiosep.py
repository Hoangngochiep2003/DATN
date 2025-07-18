from typing import Any, Callable, Dict
import random
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from models.clap_encoder import CLAP_Encoder

from huggingface_hub import PyTorchModelHubMixin

# Import các mô hình Mamba mới
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lightmunet_wrapper import LightMUNetWrapper


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        ss_model: nn.Module = None,
        waveform_mixer = None,
        query_encoder: nn.Module = CLAP_Encoder().eval(),
        loss_function = None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func = None,
        use_text_ratio: float =1.0,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model: nn.Module
            anchor_segment_detector: nn.Module
            loss_function: function or object
            learning_rate: float
            lr_lambda: function
        """

        super().__init__()
        self.ss_model = ss_model
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder
        self.query_encoder_type = self.query_encoder.encoder_type
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func


    def forward(self, x):
        pass

    def training_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. 
                'audio_text': {
                    'text': ['a sound of dog', ...]
                    'waveform': (batch_size, 1, samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        if 'audio_text' in batch_data_dict:
            batch_audio_text_dict = batch_data_dict['audio_text']
        else:
            batch_audio_text_dict = batch_data_dict

        if 'text' in batch_audio_text_dict:
            batch_text = batch_audio_text_dict['text']
        elif 'caption' in batch_audio_text_dict:
            batch_text = batch_audio_text_dict['caption']
        else:
            raise KeyError("Batch không có key 'text' hoặc 'caption'. Hãy kiểm tra lại collate_fn và dataset.")
        # batch_audio = batch_audio_text_dict['waveform']
        device = batch_audio_text_dict['waveform'].device

        # Lấy mel-spectrogram nếu có, nếu không thì dùng waveform (dành cho backward compatibility)
        mixtures = batch_audio_text_dict['mixture']
        if isinstance(mixtures, list):
            mixtures = torch.stack(mixtures, dim=0)

        # calculate text embed for audio-text data
        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='hybird',
                text=batch_text,
                audio=mixtures.squeeze(1) if mixtures.dim() > 2 else mixtures,
                use_text_ratio=self.use_text_ratio,
            )

        input_dict = {
            'mixture': mixtures,
            'condition': conditions,
        }

        # Target là waveform
        target_dict = {
            'segment': batch_audio_text_dict['waveform'],
        }

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform']
        sep_segment = sep_segment.squeeze()
        # (batch_size, 1, segment_samples) hoặc (batch_size, T, n_mels)

        output_dict = {
            'segment': sep_segment,
        }

        # Crop output và target về cùng chiều time nhỏ nhất trước khi tính loss
        output = output_dict['segment']
        target = target_dict['segment']
        # Nếu target là 4D (B, 1, T, n_mels), squeeze channel
        if target.dim() == 4:
            target = target.squeeze(1)
        # Nếu output là 4D (B, 1, T, n_mels), squeeze channel
        if output.dim() == 4:
            output = output.squeeze(1)
        min_t = min(output.shape[1], target.shape[1])
        output = output[:, :min_t, :]
        target = target[:, :min_t, :]

        # Calculate loss.
        loss = self.loss_function(output, target)

        self.log_dict({"train_loss": loss})
        
        return loss

    def validation_step(self, batch_data_dict, batch_idx):
        batch_audio_text_dict = batch_data_dict
        batch_text = batch_audio_text_dict['text']
        mixtures = batch_audio_text_dict['mixture']
        if isinstance(mixtures, list):
            mixtures = torch.stack(mixtures, dim=0)
        if self.query_encoder_type == 'CLAP':
            conditions = self.query_encoder.get_query_embed(
                modality='hybird',
                text=batch_text,
                audio=mixtures.squeeze(1) if mixtures.dim() > 2 else mixtures,
                use_text_ratio=self.use_text_ratio,
            )
        input_dict = {
            'mixture': mixtures,
            'condition': conditions,
        }
        target = batch_audio_text_dict['waveform']
        if target.dim() == 4:
            target = target.squeeze(1)
        output = self.ss_model(input_dict)['waveform']
        if output.dim() == 4:
            output = output.squeeze(1)
        min_t = min(output.shape[1], target.shape[1])
        output = output[:, :min_t, :]
        target = target[:, :min_t, :]
        # Giả sử compute_sdr là hàm đã có, nếu chưa có bạn cần định nghĩa
        sdr = compute_sdr(output, target)
        self.log('val_sdr', sdr, prog_bar=True, on_epoch=True)
        return sdr

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

        return output_dict
    

def get_model_class(model_type):
    print(f'Loading model: {model_type}')
    if model_type == 'ResUNet30' or model_type == 'DenseUNet30':
        from models.resunet import ResUNet30
        return ResUNet30
    elif model_type == 'LightMUNet':
        return LightMUNetWrapper
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")


def compute_sdr(output, target):
    # Placeholder: trả về -loss để Lightning chọn checkpoint tốt nhất
    # Bạn nên thay bằng hàm SDR thực tế nếu có
    return -torch.nn.functional.l1_loss(output, target)