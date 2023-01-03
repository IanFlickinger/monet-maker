"""PatchGAN implementation

This module contains the logic for building a tensorflow keras PatchGAN model.

TODO:
    Parameterize PatchGAN architecture by depth and receptive field
"""

from typing import Iterable

import tensorflow as tf
import tensorflow_addons as tfa


def build(
        input_shape: Iterable[int],
        dropout_rate: float = 0,
):
    """
    :param depth: int. The number of convolutional layers to stack.
    :param kernel_widths: int, Iterable[int]. The size(s) of the filters.
    If int, the model will contain depth convolutional layers, each with
    identical kernel widths as specified. If Iterable[int], their will be
    len(kernel_widths) convolutional layers with associated kernel widths.
    :param patch_size: int. The receptive field of each of the neurons in
    the final output layer.
    :param init_filters: int. The number of filters in the first convolutional
    layer. After that, filters will be halved in each layer.
    :param min_filters: int. Lower bound on number of filters in a convolutional
    layer. All layers will output at least min_filters channels except for the
    final layer (which will output one channel as the patch prediction).
    """
    #         # cast kernel_widths to tensor for consistency
    #         if type(kernel_widths) is int:
    #             kernel_widths = tf.fill((depth), kernel_widths)
    #         else:
    #             kernel_widths = tf.constant(kernel_widths)
    #             depth = kernel_widths.shape[0]

    #         # calculate stride for given patch_size, kernel_widths, and depth
    #         strides = tf.constant(kernel_widths)

    #         # reduce number of filters at each layer by factor of 2
    #         channel_depths = init_filters // (2**tf.range(depth))
    #         # apply lower bound
    #         channel_depths = tf.where(channel_depths < min_filters, min_filters, channel_depths)

    #         channel_depths=[ 64, 128, 256, 528,   1]
    #         kernel_widths= [  5,   5,   3,   3,   3]
    #         strides=       [  2,   2,   2,   2,   2]
    # final receptive field of 69x69

    channel_depths = [64, 256, 528, 1]
    kernel_widths = [5, 5, 3, 3]
    strides = [3, 3, 2, 2]
    # final receptive field of 71x71

    # build convolutional model
    layers = [tf.keras.Input(input_shape)]
    for filters, width, stride in zip(channel_depths, kernel_widths, strides):
        layers.extend([
            tf.keras.layers.SpatialDropout2D(dropout_rate),
            tf.keras.layers.Conv2D(filters, width, stride),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.Activation('leaky_relu')
        ])
    layers[-1] = tf.keras.layers.Activation('sigmoid')

    model = tf.keras.Sequential(layers)
    return model
