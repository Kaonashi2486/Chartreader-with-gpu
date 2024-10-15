import sys
# Add the path to your config.py to sys.path
sys.path.append(r'C:\Users\saksh\OneDrive\Desktop\stuffs\Chartreader-with-gpu\models\py_utils')
from py_utils import convolution
import torch.nn as nn

def make_pool_layer() -> nn.Module:
    return nn.Sequential()

#to know more about the convolution kernel click:https://medium.com/@abhishekjainindore24/all-about-convolutions-kernels-features-in-cnn-c656616390a1
# Parameter kernel: The size of the convolution kernel.
# Parameter dim0: The dimension of the input layer.
# Parameter dim1: The dimension of the output layer.
# Parameter mod: The number of modules to add, excluding the first one.
# Parameter layer: The function used to construct the layer, default is the convolution function.
# Return type nn.Module: Returns a PyTorch module.
def make_hg_layer(kernel: int, dim0: int, dim1: int, mod: int, layer=convolution, **kwargs) -> nn.Module:
    # Ensure kernel size, dim0 and dim1 are positive and mod is non-negative
    assert kernel > 0 and dim0 > 0 and dim1 > 0, "Kernel size and dimensions must be positive."
    assert mod >= 0, "Number of modules must be non-negative."
    # Use the passed-in layer function (default is the convolution function) to create the first layer with a stride of 2.
    # Store this layer in a list named layers.
    layers = [layer(kernel, dim0, dim1, stride=2)]
    # Through a loop, use the same kernel and dim1 parameters to add mod - 1 additional layers to the list layers.
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    
    return nn.Sequential(*layers)