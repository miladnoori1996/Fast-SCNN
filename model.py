# This python file implements the model for Fast-SCNN

import tensorflow as tf


def conv_layer(input_layer, conv_type, filters, kernel_size, strides, padding='same', relu=True, add_layer=None):
    """
    The function conv_Layer is an abstraction for making layers instead of
    using tf.keras.layers.?? this function supports:
        >> tf.keras.layers.Conv2D
        >> tf.keras.layers.SeparableConv2D
        >> tf.keras.layers.BatchNormalization
        >> tf.keras.activations

    :param input_layer: (type : tf.keras.layers.Input): the input to this layer
    :param conv_type: (type : string): defining the type of this convolution ("conv", "ds", "dw", "add")
    :param filters: (type : int): the number of filters/kernel
    :param kernel_size: (type : tuple): the shape of the kernel
    :param strides: (type : tuple): the stride for convolution
    :param padding: (type : string): the the padding for image
    :param relu: (type : boolean): true or false using relu activation
    :param add_layer: (type : tf.keras.layers): to add 2 layers together with conv_type "add"
    :return: (type : keras.layers.Layer)
    """
    if conv_type == 'ds':
        x = tf.keras.layers.SeparableConv2D(filters, kernel_size, strides, padding=padding)(input_layer)
    elif conv_type == 'dw':
        x = tf.keras.layers.DepthwiseConv2D(filters, strides, depth_multiplier=1, padding=padding)(input_layer)
    elif conv_type == 'add':
        x = tf.keras.layers.add([input_layer, add_layer])
    else:
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding)(input_layer)
    if relu:
        x = tf.keras.activations.relu(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def bottelneck(input_layer, filters, kernel_size, expansion_factor, n, strides):
    """
    Bottleneck function this function takes care of each bottleneck in th e paper
    each bottleneck consists of a Conv2D, DWConv, and Conv2D.
    and repeated 'n' times

    :param input_layer: (type : tf.keras.layers.Input): the input to this layer
    :param filters: (type : int): the number of filters/kernel
    :param kernel_size: (type : tuple): the shape of the kernel
    :param expansion_factor: (type : int): this number is used in the inner function
    :param n: (type : int): how many times to repeat the bottleneck
    :param strides: (type : tuple): the stride for convolution
    :return: x (type : keras.layers.Layer)
    """
    def _inner_bottelneck(input_layer, filters, kernel_size, expansion_factor, strides, add=False):
        """
        this function is used in the inner loop to continue adding the rest of the bottleneck

        :param input_layer: (type : tf.keras.layers.Input): the input to this layer
        :param filters: (type : int): the number of filters/kernel
        :param kernel_size: (type : tuple): the shape of the kernel
        :param expansion_factor: (type : int): expansion factor of the bottleneck block
        :param strides: (type : tuple): the stride for convolution
        :param add: (type : boolean): use for tf.keras.layers.add if set to True
        :return: x (type : keras.layers.Layer)
        """
        tchannel = tf.keras.backend.int_shape(input_layer)[-1] * expansion_factor
        x = conv_layer(input_layer, 'conv', tchannel, (1, 1), strides=(1, 1))
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=1, padding='same')(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = conv_layer(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)
        if add:
            x = tf.keras.layers.add([x, input_layer])
        return x
    x = _inner_bottelneck(input_layer, filters, kernel_size, expansion_factor, strides)
    for i in range(1, n):
        x = _inner_bottelneck(x, filters, kernel_size, expansion_factor, (1, 1), True)
    return x


# NOTE i did not implement this i just copied it
def ppm(input_tensor, bin_sizes):
    """
    Pyramid Pooling Block this function takes the feature map from the last convolution
    layer and applies multiple sub-region average pooling and upscaling

    :param input_tensor: (type : tensor) Feature Map
    :param bin_sizes: (type : list) different bin sizes
    :return:(type : keras.layers.Layer)
    """
    concat_list = [input_tensor]
    # for (1024, 2048)
    # w = 32
    # h = 64

    # for the new img size
    w = 8
    h = 16

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)
        concat_list.append(x)
    return tf.keras.layers.concatenate(concat_list)




