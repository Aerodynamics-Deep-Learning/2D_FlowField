"""

Here, we define the Fourier Layers that are used in the Fourier Neural Operator (FNO) architecture.

"""

import torch
import torch.nn as nn

import sys

from ..auxiliary import get_activation

#region Scripts
# Scripts for complex multiplication in different dimensions.

# These are directly taken from the paper with minor additions since it is straightfoward. Perform matmul.
@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
    res = torch.einsum("bix,iox->box", a, b)
    return res

@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    res =  torch.einsum("bixy,ioxy->boxy", a, b)
    return res


@torch.jit.script
def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res = torch.einsum("bixyt,ioxyt->boxyt", a, b)
    return res

@torch.jit.script
def compl_mul4d(a: torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,z,t), (in_channel, out_channel, x,y,z,t) -> (batch, out_channel, x,y,z,t)
    res = torch.einsum("bixyzt,ioxyzt->boxyzt", a, b)
    return res

#endregion


#region Spectral Convolution

class SpectralConvolution1D(nn.Module):
    
    """
    Performs the 1D spectral (Fourier) convolution operation.

    Args:
        in_channels: is how many inputs we have to our channel
        out_channels: is how many outputs we want
        modes: is the number of Fourier modes to keep in the layer
        norm_weights: option to normalizing weights; "paper" (1/ch1 * ch2), "geom" (1/(ch1 * ch2)**0.5), "xavier" ((2/(ch1 + ch2))**0.5)
        norm_fft: option to normalizing signal; "forward" (none), "backward" (1/n), "ortho" (1/sqrt(n))

    Returns:
        x: spectrally convoluted 1D signal
    """
    
    def __init__(self, in_channels:(int), out_channels:(int), modes:(int), norm_weights:(str)="paper", norm_fft:(str)="ortho"):
        super().__init__()

        self.in_channels = in_channels # Total number of in_channels
        self.out_channels = out_channels # Total number of out_channels
        self.norm_weights = norm_weights # Which norm to use for weight initialization
        self.norm_fft = norm_fft # Which norm to apply to the fft and ifft operations

        # How many modes of the FT we want to keep
        self.modes1 = modes

        # Initialize the weights for in-frequency-space linear transformation, the weights are complex numbers, so we use torch.cfloat
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

        # The scaling factor for the randomized weights
        # This is needed because the operations will occur on the complex space, which tends to inflate variance easily and explode the grads

        """
        "paper": The initialization found on the paper, causes the model to first improve on the local linear convolution 
        and add small spectral conv perturbations during training, since weights are insanely small initially

        "geom": Geometric scaling, which is a bit less aggressive than the paper scaling

        "xavier": Xavier initialization, which is a standard initialization for neural networks
        """

        if self.norm_weights == "paper":
            scale = 1 / (in_channels * out_channels)

        
        elif self.norm_weights == "geom":
            scale = 1 / (in_channels * out_channels)**0.5
        
        elif self.norm_weights == "xavier":
            scale = (2 / (in_channels + out_channels))**0.5
        
        else:
            raise ValueError(f"Unknown norm_weights {self.norm_weights}, please use 'paper', 'geom', or 'xavier'.")

        with torch.no_grad():
            self.weights1.copy_(
                scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
            )

        if sys.platform == 'win32':
            self.cast_in = lambda x: x.double()
            self.cast_out = lambda x: x.float()
            self.target_dtype = torch.cdouble
            self.weight_cast = lambda w: w.cdouble()
        else:
            self.cast_in = lambda x: x # Identity function (do nothing)
            self.cast_out = lambda x: x
            self.target_dtype = torch.cfloat
            self.weight_cast = lambda w: w

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Takes in a signal x (1D), applies rfft, performs spectral linear transform on selected modes, returns irfft  
        """

        batchsize = x.shape[0]
        # The first dimension is the batch size, the second is the number of channels, and the third is the signal itself on that channel.

        # Cast the input to the target dtype
        x = self.cast_in(x)

        # Compute the Fourier transform of the input using FFT
        x_ft = torch.fft.rfftn(x, dim=[2], norm=self.norm_fft) # Apply fft along the last dimension, the signal dimension. 

        # Define a zeros tensor to hold the outputs, we build it full size because we are essentiall going to assing 0 to incredibly high freqs.
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), dtype=self.target_dtype, device=x.device)
        # Perform the spectral convolution via multiplying w weights, only selecting specific modes, setting the rest equal to 0 [:,:,modes:]
        out_ft[:,:,:self.modes1] = compl_mul1d(x_ft[:,:,:self.modes1], self.weight_cast(self.weights1))

        # Compute the inverse FT using irfftn, "return to physical space", to have same outs as the size of x
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2], norm=self.norm_fft)

        return self.cast_out(x)
    
#endregion


#region Fourier Layer

class FourierLayer1D(nn.Module):

    """
    The complete Fourier Layer, as is on the original FNO paper. Take the input, apply spectral convolution, apply local linear convolution, add these two and that becomes the output.

    Args:
        in_channels: is how many inputs we have to our channel
        out_channels: is how many outputs we want
        modes: is the number of Fourier modes to keep in the layer
        kernel: kernels for the local linear convolutions
        norm_weights: option to normalizing weights; "paper" (1/ch1 * ch2), "geom" (1/(ch1 * ch2)**0.5), "xavier" ((2/(ch1 + ch2))**0.5)
        norm_fft: option to normalizing signal; "forward" (none), "backward" (1/n), "ortho" (1/sqrt(n))
    """

    def __init__(self, in_channels:(int), out_channels:(int), modes:(int), kernel:(int), norm_weights:(str), norm_fft:(str)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.kernel = kernel

        self.SpectralConv = SpectralConvolution1D(in_channels, out_channels, modes, norm_weights, norm_fft)

        self.LocalLinear = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=kernel//2)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input to the 1D Fourier layer (B, C_in, x)
        Returns:
            x: Spectrally convoluted and linear convoluted 1D output (B, C_out, x)
        """

        x1 = self.SpectralConv(x)
        x2 = self.LocalLinear(x)

        return x1 + x2

#endregion


#region MLPLayer

class MLPLayer(nn.Module):

    def __init__(self, input_dim:(int), output_dim:(int), hidden_units:(list[int]), act_fn:(str)):
        super().__init__()

        # Initialize MLP parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.act_fn = get_activation(act_fn or "gelu")

        # Add input layer
        self.layers = [nn.Linear(in_features=self.input_dim, out_features=self.hidden_units[0]), self.act_fn]

        # Add hidden layers
        for i in range(len(self.hidden_units)-1):
            self.layers.extend([
                nn.Linear(in_features=self.hidden_units[i], out_features=self.hidden_units[i+1]),
                self.act_fn
                ])

        # Add output layer
        self.layers.append(nn.Linear(in_features=self.hidden_units[-1], out_features=self.output_dim))

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):

        # Return the forward prop
        return self.mlp(input)
    
#endregion