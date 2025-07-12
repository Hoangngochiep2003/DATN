import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# === Giữ nguyên các class quan trọng từ ResUNet ===

class FiLM(nn.Module):
    def __init__(self, film_meta, condition_size):
        super(FiLM, self).__init__()
        self.condition_size = condition_size
        self.modules, _ = self.create_film_modules(film_meta=film_meta, ancestor_names=[])

    def create_film_modules(self, film_meta, ancestor_names):
        modules = {}
        for module_name, value in film_meta.items():
            if isinstance(value, int):
                ancestor_names.append(module_name)
                unique_module_name = '->'.join(ancestor_names)
                modules[module_name] = self.add_film_layer_to_module(value, unique_module_name)
            elif isinstance(value, dict):
                ancestor_names.append(module_name)
                modules[module_name], _ = self.create_film_modules(value, ancestor_names)
            ancestor_names.pop()
        return modules, ancestor_names

    def add_film_layer_to_module(self, num_features, unique_module_name):
        layer = nn.Linear(self.condition_size, num_features)
        self.add_module(name=unique_module_name, module=layer)
        return layer

    def forward(self, conditions):
        return self.calculate_film_data(conditions, self.modules)

    def calculate_film_data(self, conditions, modules):
        film_data = {}
        for module_name, module in modules.items():
            if isinstance(module, nn.Module):
                film_data[module_name] = module(conditions)[:, :, None, None]
            elif isinstance(module, dict):
                film_data[module_name] = self.calculate_film_data(conditions, module)
        return film_data


def get_film_meta(module):
    film_meta = {}
    if hasattr(module, 'has_film') and module.has_film:
        film_meta['beta1'] = module.bn1.num_features
        film_meta['beta2'] = module.bn2.num_features
    for child_name, child_module in module.named_children():
        child_meta = get_film_meta(child_module)
        if child_meta:
            film_meta[child_name] = child_meta
    return film_meta


# === WaveNet (thay thế ResUNet30) ===

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(WaveNetBlock, self).__init__()
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.residual = nn.Conv1d(out_channels, in_channels, 1)
        self.skip = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        z = filter_out * gate_out
        skip = self.skip(z)
        res = self.residual(z) + x
        return res, skip


class WaveNet_Base(nn.Module):
    def __init__(self, input_channels, output_channels, num_blocks=6, channels=64):
        super(WaveNet_Base, self).__init__()
        self.input_conv = nn.Conv1d(input_channels, channels, kernel_size=1)
        self.wavenet_blocks = nn.ModuleList()
        for b in range(num_blocks):
            dilation = 2 ** b
            self.wavenet_blocks.append(WaveNetBlock(channels, channels, kernel_size=3, dilation=dilation))
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, output_channels, kernel_size=1)
        )

    def forward(self, mixtures, film_dict=None):
        x = self.input_conv(mixtures)  # (B, C, T)
        skip_connections = []
        for block in self.wavenet_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        out = sum(skip_connections)
        out = self.output_conv(out)
        return {'waveform': out}



# class WaveNet(nn.Module):
#     def __init__(self, input_channels, output_channels, condition_size):
#         super(WaveNet, self).__init__()
#         self.base = WaveNet_Base(input_channels=input_channels, output_channels=output_channels)
#         self.film_meta = get_film_meta(self.base)
#         self.film = FiLM(film_meta=self.film_meta, condition_size=condition_size)

#     def forward(self, input_dict):
#         mixtures = input_dict['mixture']  # (B, 1, T)
#         conditions = input_dict['condition']
#         film_dict = self.film(conditions)
#         return self.base(mixtures, film_dict)

#     @torch.no_grad()
#     def chunk_inference(self, input_dict):
#         chunk_config = {'NL': 1.0, 'NC': 3.0, 'NR': 1.0, 'RATE': 16000}
#         mixtures = input_dict['mixture']
#         conditions = input_dict['condition']
#         film_dict = self.film(conditions)

#         NL = int(chunk_config['NL'] * chunk_config['RATE'])
#         NC = int(chunk_config['NC'] * chunk_config['RATE'])
#         NR = int(chunk_config['NR'] * chunk_config['RATE'])
#         L = mixtures.shape[2]

#         out_np = torch.zeros((1, L), device=mixtures.device)
#         WINDOW = NL + NC + NR
#         current_idx = 0

#         while current_idx + WINDOW < L:
#             chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
#             chunk_out = self.base(chunk_in, film_dict=film_dict)['waveform']

#             if current_idx == 0:
#                 out_np[:, current_idx:current_idx + WINDOW - NR] = chunk_out[:, :-NR] if NR != 0 else chunk_out
#             else:
#                 out_np[:, current_idx + NL:current_idx + WINDOW - NR] = chunk_out[:, NL:-NR] if NR != 0 else chunk_out[:, NL:]

#             current_idx += NC

#         return out_np.cpu().numpy()


class WaveNet(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size):
        super(WaveNet, self).__init__()
        self.base = WaveNet_Base(input_channels=input_channels, output_channels=output_channels)
        self.film_meta = get_film_meta(self.base)
        self.film = FiLM(film_meta=self.film_meta, condition_size=condition_size)

    def forward(self, input_dict):
        mixtures = input_dict['mixture']       # (B, 1, T)
        conditions = input_dict['condition']   # (B, D) từ CLAP
        film_dict = self.film(conditions)
        return self.base(mixtures, film_dict)

    @torch.no_grad()
    def chunk_inference(self, input_dict):
        chunk_config = {'NL': 1.0, 'NC': 3.0, 'NR': 1.0, 'RATE': 16000}
        mixtures = input_dict['mixture']       # (B, 1, T)
        conditions = input_dict['condition']   # (B, D)
        film_dict = self.film(conditions)

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])
        L = mixtures.shape[2]

        out_np = torch.zeros((1, L), device=mixtures.device)
        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
            chunk_out = self.base(chunk_in, film_dict=film_dict)['waveform']

            if current_idx == 0:
                out_np[:, current_idx:current_idx + WINDOW - NR] = chunk_out[:, :-NR] if NR != 0 else chunk_out
            else:
                out_np[:, current_idx + NL:current_idx + WINDOW - NR] = chunk_out[:, NL:-NR] if NR != 0 else chunk_out[:, NL:]

            current_idx += NC

        return out_np.cpu().numpy()
