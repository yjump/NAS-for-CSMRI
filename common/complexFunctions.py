#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: spopoff
"""
from torch.nn.functional import relu, max_pool2d, leaky_relu, interpolate


def complex_relu(input_r, input_i):
    #    assert(input_r.size() == input_i.size())
    return relu(input_r), relu(input_i)


def complex_leakyrelu(input_r, input_i):
    return leaky_relu(input_r), leaky_relu(input_i)


def complex_upsample(input_r, input_i, factor):
    return interpolate(input_r, scale_factor=factor, mode='bilinear', align_corners=False), interpolate(input_i, scale_factor=factor, mode='bilinear', align_corners=False)


def complex_max_pool2d(input_r, input_i, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):
    return max_pool2d(input_r, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices), \
           max_pool2d(input_i, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices)
