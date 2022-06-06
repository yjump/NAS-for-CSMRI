import math
import torch
import torch.nn as nn
import common.fftpytorch as myfft
from models.dcnet.rdnet import RDBlock


def get_padding_layer(total_padding, mode='zero'):
    padding_layers = {
        'zero': nn.ZeroPad2d,
        'reflection': nn.ReflectionPad2d,
        'replication': nn.ReplicationPad2d
    }
    assert mode in padding_layers

    padding_side = total_padding // 2
    if total_padding % 2 == 0:
        padding = padding_side
    else:
        padding = (padding_side, padding_side + 1, padding_side, padding_side + 1)

    return padding_layers[mode](padding)


def get_same_padding_layer(kernel_size, stride, mode='zero', dilation=1):
    """Constructs padding layer for SAME padding

    Calculates padding to insert such that the spatial dimensions stay the same
    after a 2d convolution.
    WARNING: Only works for even sized input sizes and stride one or two.
    """
    assert stride == 1 or stride == 2, 'Formula only works for stride 1 or 2'
    effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    total_padding = int(math.ceil((effective_kernel_size - 1.0) / stride))
    return get_padding_layer(total_padding, mode)


class ConvBlock(nn.Module):
    def __init__(self, num_convs, num_filters, kernel_size, relu_leakiness,
                 dilations, padding='zero', num_inputs=2, num_outputs=2,
                 final_act=False):
        super(ConvBlock, self).__init__()
        in_channels = num_inputs

        modules = []
        for i in range(num_convs - 1):
            modules += [get_same_padding_layer(kernel_size, stride=1,
                                               mode=padding, dilation=dilations[i]),
                        nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size,
                                  stride=1, bias=True, dilation=dilations[i]),
                        nn.LeakyReLU(relu_leakiness, inplace=True)]
            in_channels = num_filters

        modules += [get_same_padding_layer(kernel_size, stride=1,
                                           mode=padding, dilation=dilations[-1]),
                    nn.Conv2d(in_channels, num_outputs, kernel_size=kernel_size,
                              stride=1, bias=True, dilation=dilations[-1])]
        if final_act:
            modules.append(nn.LeakyReLU(relu_leakiness, inplace=True))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class RecNet(nn.Module):
    """Reconstruction Network

    After Schlemper et al: A Deep Cascade of Convolutional Neural Networks for
    Dynamic MR Image Reconstruction
    """
    DEFAULT_RELU_LEAKINESS = 0.01

    def __init__(self, num_blocks=3, num_convs=3, num_filters=32,
                 num_final_outputs=2, dilations_per_conv=1, kernel_size=3,
                 relu_leakiness=DEFAULT_RELU_LEAKINESS, padding='zero',
                 use_refinement=False, skip_final_dc=False,
                 return_intermediate_recs=False):
        """Build a RecNet

        Parameters
        ----------
        num_blocks : int
          Number of data consistency blocks. Each data consistency block is
          a number of convolutional layers followed by a data consistency layer
        num_convs : int
          Number of convolutional layers per data consistency block
        num_filters : int or list
          Number of filters per convolutional layer. If list, uses specified filter
          size for each data consistency block
        num_final_outputs : int
          Number of output channels to produce after the final block
        dilations_per_conv : int or list
          If given, sets the dilation rate of convolutions within a block
        kernel_size : int
          Convolutional filter size
        relu_leakiness : float
          Leakiness of lrelu
        padding : string
          Type of padding to use. Either `zero`, `reflection`, or `replication`
        use_refinement : bool
          If true, learn an additive transformation with respect to the input
          image
        skip_final_dc : bool
          If true, skips the final data consistency layer
        return_intermediate_recs : bool
          If true, returns intermediate reconstructions under key `reconstructions`
        """
        super(RecNet, self).__init__()
        if isinstance(num_filters, int):
            num_filters = [num_filters] * num_blocks
        if isinstance(dilations_per_conv, int):
            dilations_per_conv = [dilations_per_conv] * num_convs

        assert len(num_filters) == num_blocks, \
            'Number of given filters must match number of blocks'
        assert len(dilations_per_conv) == num_convs, \
            'Number of dilations must match number of convolutions'

        conv_blocks = []
        for idx, num_filter in enumerate(num_filters):
            num_outputs = 2 if idx < num_blocks - 1 else num_final_outputs
            # conv_blocks.append(ConvBlock(num_convs, num_filter,
            #                              kernel_size, relu_leakiness,
            #                              padding=padding,
            #                              num_outputs=num_outputs,
            #                              dilations=dilations_per_conv))
            conv_blocks.append(RDBlock(in_channels=2, out_channels=2))

        dc_layers = []
        num_dc_layers = num_blocks if not skip_final_dc else num_blocks - 1
        for i in range(num_dc_layers):
            dc_layers.append(myfft.DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_layers = dc_layers
        self.use_refinement = use_refinement
        self.skip_final_dc = skip_final_dc
        self.return_intermediate_recs = return_intermediate_recs

    def forward(self, inp, kspace, mask, image_mod):
    # def forward(self, inp, kspace, mask, mean, std):
        x = inp
        reconstructions = []
        for idx in range(len(self.conv_blocks)):
            block_input = x

            x = self.conv_blocks[idx](x)

            if self.use_refinement:
                x = x + block_input

            if idx < len(self.dc_layers):
                x = x * image_mod / 6.0
                # x = x * std + mean
                x = self.dc_layers[idx].perform(x, kspace, mask)
                x = x * 6.0 / image_mod
                # x = (x - mean) / std
                if self.return_intermediate_recs:
                    reconstructions.append(x)

        if self.return_intermediate_recs:
            return {
                'pred': x,
                'reconstructions': reconstructions
            }
        else:
            return x
