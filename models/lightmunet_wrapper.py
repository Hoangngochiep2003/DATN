import torch
import torch.nn as nn
import sys
import os

# Import LightMUNet từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LightMUNet import LightMUNet

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
        
    def _create_film_layers(self, condition_size):
        """Tạo FiLM layers cho conditioning"""
        film_layers = nn.ModuleDict()
        
        # FiLM cho convInit
        film_layers['convInit'] = nn.Linear(condition_size, self.init_filters * 2)
        
        # FiLM cho down layers
        for i, blocks in enumerate([1, 2, 2, 4]):
            layer_channels = self.init_filters * (2 ** i)
            film_layers[f'down_{i}'] = nn.Linear(condition_size, layer_channels * 2)
            
        # FiLM cho up layers
        for i, blocks in enumerate([1, 1, 1]):
            layer_channels = self.init_filters * (2 ** (2 - i))
            film_layers[f'up_{i}'] = nn.Linear(condition_size, layer_channels * 2)
            
        # FiLM cho final conv
        film_layers['final'] = nn.Linear(condition_size, self.init_filters * 2)
        
        return film_layers
    
    def _apply_film(self, x, condition, layer_name):
        """Áp dụng FiLM conditioning"""
        if layer_name in self.film_layers:
            film_params = self.film_layers[layer_name](condition)
            gamma, beta = film_params.chunk(2, dim=1)
            
            # Reshape để broadcast
            if len(x.shape) == 4:  # (B, C, H, W)
                gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
                beta = beta.view(beta.size(0), beta.size(1), 1, 1)
            
            x = gamma * x + beta
        return x
    
    def forward(self, input_dict):
        """
        Forward pass tương thích với ResUNet30 interface
        """
        mixtures = input_dict['mixture']  # (B, 1, T, F)
        conditions = input_dict['condition']  # (B, condition_size)
        
        # Áp dụng FiLM cho convInit
        x = self._apply_film(mixtures, conditions, 'convInit')
        x = self.lightmunet.convInit(x)
        
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
            x = x + down_x[i + 1]  # Skip connection
            x = self._apply_film(x, conditions, f'up_{i}')
            x = up_layer(x)
        
        # Final conv với FiLM
        x = self._apply_film(x, conditions, 'final')
        x = self.lightmunet.conv_final(x)
        
        # Output format tương thích
        output_dict = {'waveform': x}
        return output_dict 