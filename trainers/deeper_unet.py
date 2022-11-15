"""deeper_unet.py

A custom U-Net implementation in PyTorch
"""

from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
import torchvision.transforms
import numpy as np


class EncoderBlock(nn.Module):
    """U-Net encoder block"""

    def __init__(self, in_ch: int, out_ch: int):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.activation = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class BottleneckBlock(nn.Module):
    "The U-Net bottleneck layer, with dropouts"

    def __init__(self, in_ch: int, out_ch: int):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.activation = nn.LeakyReLU()
        # self.dropout = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        return x   


class DecoderBlock(nn.Module):
    """U-Net decoder block with skip connections.

    Takes 2N channels from the previous layer, up-convolves to N channels at double size. Concatenates with 
    skip-connection's encoder-features (of N channels) to get 2N channels. Passes through two convolutional layers to 
    get N channels as the output. Hence in_channels and out_channels should correspond to 2N and N.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.activation = nn.LeakyReLU()

    def forward(self, x, skip_features):
        x = self.upconv(x)  # Converts input from 2N -> N channels
        skip_features = self.crop(skip_features, x)  # Crops the skip features
        x = torch.cat((skip_features, x), dim=1)  # Stacks (N+N) channels
        x = self.conv1(x)  # 2N channels -> N channels
        x = self.activation(x)
        x = self.conv2(x)  # N channels -> N channels
        x = self.activation(x)
        return x

    def crop(self, encoder_features, x):
        """Crops the encoder's features to the shape of the input to the block"""
        _, _, H, W = x.shape
        encoder_features = torchvision.transforms.CenterCrop((H, W))(encoder_features)
        return encoder_features


class HeadBlock(nn.Module):
    """U-Net head, converting the output of the decoder into the desired number of channels with 1D convolutions"""

    def __init__(self, in_ch: int, out_ch: int):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, (1, 1))
        self.activation = nn.Softmax()  #nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class UNet(nn.Module):
    """An implementation of U-Net with skip connections.

    Don't forget to send the network to the device with net.to(device) and same for the data it operates on with 
    tensor.to(device)!
    """

    def __init__(self, input_channels: int, out_channels: int, first_layer_channels: int, num_layers: int, 
                 input_image_shape: Tuple[int, int]):
        """Defines the U-Net. Specify the desired input and output dimensions and the desired output dimensions of the
        Encoder backbone.

        Parameters
        ----------
        input_channels : int
            The number of channels in the network's input data
        out_channels : int
            The number of channels in the network's output data
        first_layer_channels : int
            The number of channels formed in the first layer of the network. This is doubled in every subsequent layer 
            of the encoder.
        num_layers : int
            The depth of the network i.e. the number of layers in the encoder branch of the network.
        input_image_shape : Tuple[int, int]
            The input image shape in (H, W). The output will be interpolated to match this.
        """

        super(UNet, self).__init__()

        self.input_image_shape = input_image_shape

        inner_channels = [first_layer_channels * 2**layer for layer in range(num_layers)]

        encoder_block_input_channels = [input_channels,] + inner_channels[:-1]
        encoder_block_output_channels = inner_channels
        
        bottleneck_block_input_channels = encoder_block_output_channels[-1]
        bottleneck_block_output_channels = bottleneck_block_input_channels * 2

        decoder_block_input_channels = [bottleneck_block_output_channels,] + list(reversed(inner_channels))[:-1]
        decoder_block_output_channels = list(reversed(inner_channels))

        head_block_input_channels = decoder_block_output_channels[-1]
        head_block_output_channels = out_channels


        self.encoder_blocks = []

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(in_ch, out_ch) for in_ch, out_ch
            in zip(encoder_block_input_channels, encoder_block_output_channels)
            ])

        self.bottleneck_block = BottleneckBlock(bottleneck_block_input_channels, bottleneck_block_output_channels) 

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch) for in_ch, out_ch 
            in zip(decoder_block_input_channels, decoder_block_output_channels)
            ])

        self.head_block = HeadBlock(head_block_input_channels, head_block_output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the batch forwards through the network."""
        encoder_features = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_features.append(x)  # For the skip connections

        x = self.bottleneck_block(x)

        for decoder_block, encoded_feature in zip(self.decoder_blocks, reversed(encoder_features)):
            x = decoder_block(x, encoded_feature)

        x = self.head_block(x)

        x = nn.functional.interpolate(x, self.input_image_shape, mode='bilinear')

        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes a prediction from the trained model.
        
        Makes a prediction from the trained model. Expects an image in the format (frame, channel, height, width). 
        Where there are fewer dimensions it will be assumed that the frame, then channel, were squeezed/flattened. Do 
        not for get to ensure the input data and the model are on the same device!
        """

        self.eval()  # Do not compute the gradient in the forward pass

        in_dims = len(x.shape)
        if in_dims == 2:
            x = x[None, None, ...]
        elif in_dims == 3:  # Add frame_num dimension
            x = x[None, ...]
        
        x = torch.from_numpy(x)
        output = self.forward(x)
        output = output.detach().numpy()

        if in_dims == 2:
            return output[0]
        return output

    def save(self, filename: Path):
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(cls, filename: Path):
        return cls().load_state_dict(torch.load(filename))

    @property
    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters())
