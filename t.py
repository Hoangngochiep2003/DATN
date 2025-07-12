import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        super(ConvBlockRes, self).__init__()
        
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1), 
                              padding=(kernel_size[0]//2, kernel_size[1]//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=(1, 1), 
                              padding=(kernel_size[0]//2, kernel_size[1]//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels, momentum=momentum)
        else:
            self.shortcut = None
            
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=0.1)
            
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            residual = self.bn_shortcut(residual)
            
        x += residual
        
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=0.1)
            
        return x

class EncoderBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes, self).__init__()
        
        self.conv_block = ConvBlockRes(in_channels, out_channels, kernel_size=(3, 3),
                                      activation=activation, momentum=momentum)
        self.downsample = downsample
        
    def forward(self, x):
        x = self.conv_block(x)
        x_pool = F.avg_pool2d(x, kernel_size=self.downsample)
        return x_pool, x

class DecoderBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes, self).__init__()
        
        self.activation = activation
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                       output_padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv_block = ConvBlockRes(out_channels * 2, out_channels, kernel_size=(3, 3),
                                      activation=activation, momentum=momentum)
        
    def forward(self, x, skip_connection):
        x = self.conv1(x)
        x = self.bn1(x)
        
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=0.1)
            
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv_block(x)
        return x

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)

class ResUNet(nn.Module):
    def __init__(self, channels, nsrc=1):
        super(ResUNet, self).__init__()
        activation = 'relu'
        momentum = 0.01

        self.nsrc = nsrc
        self.channels = channels
        self.downsample_ratio = 2 ** 6  # 2^{#encoder_blocks}

        self.encoder_block1 = EncoderBlockRes(in_channels=channels * nsrc, out_channels=32,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes(in_channels=32, out_channels=64,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes(in_channels=64, out_channels=128,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes(in_channels=128, out_channels=256,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlockRes(in_channels=256, out_channels=384,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlockRes(in_channels=384, out_channels=384,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = ConvBlockRes(in_channels=384, out_channels=384,
                                       kernel_size=(3, 3), activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlockRes(in_channels=384, out_channels=384,
                                             stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlockRes(in_channels=384, out_channels=384,
                                             stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockRes(in_channels=384, out_channels=256,
                                             stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes(in_channels=256, out_channels=128,
                                             stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes(in_channels=128, out_channels=64,
                                             stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes(in_channels=64, out_channels=32,
                                             stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = ConvBlockRes(in_channels=32, out_channels=32,
                                            kernel_size=(3, 3), activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=1,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, sp):
        """
        Args:
          input: sp: (batch_size, channels_num, time_steps, freq_bins)
        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """

        x = sp
        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0: x.shape[-1] - 2]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T/2, F/2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T/4, F/4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T/8, F/8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T/16, F/16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 384, T/32, F/32)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 384, T/64, F/64)
        x_center = self.conv_block7(x6_pool)  # (bs, 384, T/64, F/64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T/32, F/32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T/16, F/16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T/8, F/8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T/4, F/4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T/2, F/2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, 1, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 2))
        x = x[:, :, 0: origin_len, :]
        return x

# Thêm tính năng FiLM conditioning
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super(FiLMLayer, self).__init__()
        self.film = nn.Linear(cond_dim, feature_dim * 2)
        
    def forward(self, x, cond):
        film_params = self.film(cond).unsqueeze(2).unsqueeze(3)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        return gamma * x + beta

class ResUNet_FiLM(nn.Module):
    def __init__(self, channels, cond_embedding_dim, nsrc=1):
        super(ResUNet_FiLM, self).__init__()
        activation = 'relu'
        momentum = 0.01

        self.nsrc = nsrc
        self.channels = channels
        self.downsample_ratio = 2 ** 6  # 2^{#encoder_blocks}
        self.base_model = ResUNet(channels, nsrc)
        
        # FiLM conditioning layers
        self.film1 = FiLMLayer(32, cond_embedding_dim)
        self.film2 = FiLMLayer(64, cond_embedding_dim)
        self.film3 = FiLMLayer(128, cond_embedding_dim)
        self.film4 = FiLMLayer(256, cond_embedding_dim)
        self.film5 = FiLMLayer(384, cond_embedding_dim)
        self.film6 = FiLMLayer(384, cond_embedding_dim)
        self.film7 = FiLMLayer(384, cond_embedding_dim)
        
    def forward(self, sp, cond_vec):
        """
        Args:
          sp: (batch_size, channels_num, time_steps, freq_bins)
          cond_vec: (batch_size, cond_embedding_dim)
        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        x = sp
        # Pad spectrogram to be evenly divided by downsample ratio
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0: x.shape[-1] - 2]  # (bs, channels, T, F)

        # Get the base model components but apply FiLM conditioning
        (x1_pool, x1) = self.base_model.encoder_block1(x)
        x1_pool = self.film1(x1_pool, cond_vec)
        
        (x2_pool, x2) = self.base_model.encoder_block2(x1_pool)
        x2_pool = self.film2(x2_pool, cond_vec)
        
        (x3_pool, x3) = self.base_model.encoder_block3(x2_pool)
        x3_pool = self.film3(x3_pool, cond_vec)
        
        (x4_pool, x4) = self.base_model.encoder_block4(x3_pool)
        x4_pool = self.film4(x4_pool, cond_vec)
        
        (x5_pool, x5) = self.base_model.encoder_block5(x4_pool)
        x5_pool = self.film5(x5_pool, cond_vec)
        
        (x6_pool, x6) = self.base_model.encoder_block6(x5_pool)
        x6_pool = self.film6(x6_pool, cond_vec)
        
        x_center = self.base_model.conv_block7(x6_pool)
        x_center = self.film7(x_center, cond_vec)
        
        # Decoder path (without FiLM for simplicity)
        x7 = self.base_model.decoder_block1(x_center, x6)
        x8 = self.base_model.decoder_block2(x7, x5)
        x9 = self.base_model.decoder_block3(x8, x4)
        x10 = self.base_model.decoder_block4(x9, x3)
        x11 = self.base_model.decoder_block5(x10, x2)
        x12 = self.base_model.decoder_block6(x11, x1)
        x = self.base_model.after_conv_block1(x12)
        x = self.base_model.after_conv2(x)

        # Recover shape
        x = F.pad(x, pad=(0, 2))
        x = x[:, :, 0: origin_len, :]
        return x

if __name__ == "__main__":
    # Test basic ResUNet
    model1 = ResUNet(channels=1)
    output1 = model1(torch.randn((1, 1, 1001, 513)))
    print("ResUNet output shape:", output1.size())
    
    # Test ResUNet with FiLM conditioning
    model2 = ResUNet_FiLM(channels=1, cond_embedding_dim=16)
    cond_vec = torch.randn((1, 16))
    output2 = model2(torch.randn((1, 1, 1001, 513)), cond_vec)
    print("ResUNet_FiLM output shape:", output2.size())