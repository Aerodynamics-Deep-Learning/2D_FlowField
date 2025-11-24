
#region Imports

import torch.nn as nn

from .FNO_Components import *

#endregion

#region FNO Class(es)

class FNO1D(nn.Module):

    def __init__(self, cfg_p:(dict), cfg_q:(dict), fb_hidden_dim:(list[int]), fb_modes:(list[int]), fb_kernel:(list[int]), fb_act_fn:(str), fb_norm_weights:(str), fb_norm_fft:(str)):
        super().__init__()

        # Initialize P Network (elevator)
        self.P_Net = MLPLayer(**cfg_p)

        # Initialize Q Network (reducer)
        self.Q_Net = MLPLayer(**cfg_q)

        # Initialize Fourier Block
        self.fb_in = cfg_p['output_dim']
        self.fb_out = cfg_q['input_dim']
        self.fb_hidden_dim = fb_hidden_dim
        self.fb_modes = fb_modes
        self.fb_kernel = fb_kernel
        self.fb_norm_weights = fb_norm_weights
        self.fb_norm_fft = fb_norm_fft
        self.fb_activation = get_activation(fb_act_fn or "gelu")

        assert len(self.fb_kernel) == len(self.fb_modes), 'Fourier Block kernel and modes lists have to be same len.'
        assert len(self.fb_hidden_dim) + 1 == len(self.fb_kernel), 'Hidden list has to be 1 less than kernel list in ken.'

        # Build Fourier Block, init block
        self.FourierBlock = [
            FourierLayer1D(
                in_channels= self.fb_in,
                out_channels= self.fb_hidden_dim[0],
                modes = self.fb_modes[0],
                kernel= self.fb_kernel[0],
                norm_weights= self.fb_norm_weights,
                norm_fft= self.fb_norm_fft
            ),
            self.fb_activation
        ]

        # Additional blocks
        for i in range(len(self.fb_hidden_dim) - 1):

            self.FourierBlock.extend([
                FourierLayer1D(
                    in_channels= self.fb_hidden_dim[i],
                    out_channels= self.fb_hidden_dim[i+1],
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
                    in_channels= self.fb_hidden_dim[-1],
                    out_channels= self.fb_out,
                    modes= self.fb_modes[-1],
                    kernel= self.fb_kernel[-1],
                    norm_weights= self.fb_norm_weights,
                    norm_fft= self.fb_norm_fft
                ),
                self.fb_activation
            ])

        self.FourierBlock = nn.Sequential(*self.FourierBlock)

    def forward(self, input):

        # Put it through the p network first
        input = self.P_Net(input)

        # Then through the fourier blocks
        input = self.FourierBlock(input)

        # Lastly through the q network, return the result
        return self.Q_Net(input)

#endregion
