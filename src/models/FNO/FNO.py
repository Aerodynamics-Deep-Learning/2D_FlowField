
#region Imports

import torch.nn as nn

from .FNO_Components import *

#endregion


#region FNO Class(es)

class FNO1D(nn.Module):

    """
    1 Dimensional Fourier Neural Operator. Elevates channel dim using P net, applies Fourier Blocks, lowers channel dim using Q net.

    Args:
        cfg_p (dict): Config for the p network, check details at ChannelMLP class
        cfg_q (dict): Config for the q network, check details at CHannelMLP Class
        fb_hidden_channels (list[int]): List of hidden channels for the Fourier Block layers
        fb_modes (list[int]): List of modes for the Fourier Block layers
        fb_kernel (list[int]): List of kernel sizes for the Fourier Block layers
        fb_act_fn (str): Activation function for the Fourier Block layers
        fb_norm_weights (str): Normalization type for the Fourier Block weights
        fb_norm_fft (str): Normalization type for the Fourier Block FFT
    """

    def __init__(self, cfg_p:(dict), cfg_q:(dict), fb_hidden_channels:(list[int]), fb_modes:(list[int]), fb_kernel:(list[int]), fb_act_fn:(str), fb_norm_weights:(str), fb_norm_fft:(str)):
        super().__init__()

        # Initialize P Network (elevator)
        self.P_Net = ChannelMLP(**cfg_p)

        # Initialize Q Network (reducer)
        self.Q_Net = ChannelMLP(**cfg_q)

        # Initialize Fourier Block
        self.fb_in = cfg_p['out_channels']
        self.fb_out = cfg_q['in_channels']
        self.fb_hidden_channels = fb_hidden_channels
        self.fb_modes = fb_modes
        self.fb_kernel = fb_kernel
        self.fb_norm_weights = fb_norm_weights
        self.fb_norm_fft = fb_norm_fft
        self.fb_activation = get_activation(fb_act_fn or "gelu")

        assert len(self.fb_kernel) == len(self.fb_modes), 'Fourier Block kernel and modes lists have to be same len.'
        assert len(self.fb_hidden_channels) + 1 == len(self.fb_kernel), 'Hidden channel list has to be 1 less than kernel list in ken.'

        # Build Fourier Block, init block
        self.FourierBlock = [
            FourierLayer1D(
                in_channels= self.fb_in,
                out_channels= self.fb_hidden_channels[0],
                modes = self.fb_modes[0],
                kernel= self.fb_kernel[0],
                norm_weights= self.fb_norm_weights,
                norm_fft= self.fb_norm_fft
            ),
            self.fb_activation
        ]

        # Additional blocks
        for i in range(len(self.fb_hidden_channels) - 1):

            self.FourierBlock.extend([
                FourierLayer1D(
                    in_channels= self.fb_hidden_channels[i],
                    out_channels= self.fb_hidden_channels[i+1],
                    modes= self.fb_modes[i+1],
                    kernel= self.fb_kernel[i+1],
                    norm_weights= self.fb_norm_weights,
                    norm_fft= self.fb_norm_fft
                ),
                self.fb_activation
            ])

        # Last block
        self.FourierBlock.extend([
                FourierLayer1D(
                    in_channels= self.fb_hidden_channels[-1],
                    out_channels= self.fb_out,
                    modes= self.fb_modes[-1],
                    kernel= self.fb_kernel[-1],
                    norm_weights= self.fb_norm_weights,
                    norm_fft= self.fb_norm_fft
                ),
                self.fb_activation
            ])

        self.FourierBlock = nn.Sequential(*self.FourierBlock)

    def forward(self, input:(torch.tensor)):

        """
        Forward prop for the FNO1D class. Elevates channel dim using P net, applies Fourier Blocks, lowers channel dim using Q net.
        
        Args:
            input: An input of size (Batch, In_Channels, x)

        Returns:
            output: An output of size (Batch, Out_Channels, x)
        """

        # Put it through the p network first
        input = self.P_Net(input)

        # Then through the fourier blocks
        input = self.FourierBlock(input)

        # Lastly through the q network, return the result
        return self.Q_Net(input)

#endregion
