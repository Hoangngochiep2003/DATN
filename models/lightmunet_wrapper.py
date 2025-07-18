import torch
import torch.nn as nn
import sys
import os

# Import LightMUNet từ models.LightMUNet
from models.LightMUNet import LightMUNet

class LightMUNetWrapper(nn.Module):
    """
    Wrapper cho LightMUNet để có interface tương thích với ResUNet30
    """
    def __init__(self, input_channels, output_channels, condition_size):
        super().__init__()
        
        # LightMUNet parameters
        self.spatial_dims = 2  # 2D cho audio spectrogram
        self.init_filters = 32  # Số filter ban đầu
        self.in_channels = input_channels
        self.out_channels = output_channels
        
        # Tạo LightMUNet model
        self.lightmunet = LightMUNet(
            spatial_dims=self.spatial_dims,
            init_filters=self.init_filters,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            dropout_prob=0.1,
            act=("RELU", {"inplace": True}),
            norm=("GROUP", {"num_groups": 8}),
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
        
        # FiLM layers cho conditioning (tương tự ResUNet30)
        self.film_layers = self._create_film_layers(condition_size)
        
    def _get_last_out_channels(self, layer):
        # Nếu là container (nn.Sequential), lấy block cuối cùng có out_channels
        if hasattr(layer, '__iter__') and not isinstance(layer, nn.Linear):
            for sub in reversed(list(layer)):
                if hasattr(sub, 'out_channels'):
                    return sub.out_channels
        # Nếu là module đơn lẻ
        if hasattr(layer, 'out_channels'):
            return layer.out_channels
        return None

    def _create_film_layers(self, condition_size):
        """Tạo FiLM layers cho conditioning"""
        film_layers = nn.ModuleDict()
        # FiLM cho convInit
        film_layers['convInit'] = nn.Linear(condition_size, self.init_filters * 2)
        # FiLM cho down layers (lấy số channels thực tế từ block cuối cùng)
        for i, down_layer in enumerate(self.lightmunet.down_layers):
            layer_channels = self._get_last_out_channels(down_layer)
            if layer_channels is None:
                layer_channels = self.init_filters * (2 ** i)
            film_layers[f'down_{i}'] = nn.Linear(condition_size, layer_channels * 2)
        # FiLM cho up layers (lấy số channels thực tế từ block cuối cùng)
        for i, up_layer in enumerate(self.lightmunet.up_layers):
            layer_channels = self._get_last_out_channels(up_layer)
            if layer_channels is None:
                layer_channels = self.init_filters * (2 ** (2 - i))
            film_layers[f'up_{i}'] = nn.Linear(condition_size, layer_channels * 2)
        # FiLM cho final conv
        film_layers['final'] = nn.Linear(condition_size, self.init_filters * 2)
        return film_layers

    def _apply_film(self, x, condition, layer_name):
        """Áp dụng FiLM conditioning với FiLM layer động theo shape thực tế của x"""
        def safe_int(val):
            try:
                if hasattr(val, 'item'):
                    return int(val.item())
                return int(val)
            except Exception:
                return 1  # fallback an toàn
        in_features = safe_int(condition.shape[1])
        ch = safe_int(x.shape[1])
        out_features = ch * 2
        assert isinstance(in_features, int), f"in_features must be int, got {type(in_features)}"
        assert isinstance(out_features, int), f"out_features must be int, got {type(out_features)}"
        # Nếu chưa có FiLM layer phù hợp, tạo mới
        if (layer_name not in self.film_layers) or (self.film_layers[layer_name].out_features != out_features):
            self.film_layers[layer_name] = nn.Linear(in_features, out_features).to(x.device)
        film_params = self.film_layers[layer_name](condition)
        gamma, beta = film_params.chunk(2, dim=1)
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        if gamma.shape[1] != x.shape[1]:
            print(f"[FiLM shape mismatch] layer: {layer_name}, x.shape: {x.shape}, gamma.shape: {gamma.shape}")
        x = gamma * x + beta
        return x
    
    def forward(self, input_dict):
        """
        Forward pass tương thích với ResUNet30 interface
        """
        mixtures = input_dict['mixture']  # (B, 1, T, F) mong muốn
        conditions = input_dict['condition']  # (B, condition_size)

        # Đảm bảo mixtures luôn là [B, 1, T, F]
        if mixtures.dim() == 3:
            mixtures = mixtures.unsqueeze(1)
        elif mixtures.shape[1] != 1:
            # Nếu channel != 1, lấy channel đầu tiên
            mixtures = mixtures[:, 0:1, ...]

        print("mixtures shape:", mixtures.shape)

        x = self.lightmunet.convInit(mixtures)
        x = self._apply_film(x, conditions, 'convInit')

        # Down sampling với FiLM
        down_x = []
        for i, down_layer in enumerate(self.lightmunet.down_layers):
            x = self._apply_film(x, conditions, f'down_{i}')
            x = down_layer(x)
            down_x.append(x)

        # Up sampling với FiLM
        down_x.reverse()
        for i, (up_sample, up_layer) in enumerate(zip(self.lightmunet.up_samples, self.lightmunet.up_layers)):
            x = up_sample(x)
            # Crop để skip connection cùng shape
            skip = down_x[i + 1]
            # Crop từng chiều về min
            min_shape = [min(x.shape[d], skip.shape[d]) for d in range(x.dim())]
            x = x[..., :min_shape[-2], :min_shape[-1]]
            skip = skip[..., :min_shape[-2], :min_shape[-1]]
            x = x + skip  # Skip connection
            x = self._apply_film(x, conditions, f'up_{i}')
            x = up_layer(x)

        # Final conv với FiLM
        x = self._apply_film(x, conditions, 'final')
        x = self.lightmunet.conv_final(x)

        # Output format tương thích
        output_dict = {'waveform': x}
        return output_dict 