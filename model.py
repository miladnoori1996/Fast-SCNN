# This python file implements the model for Fast-SCNN
import os
import tensorflow as tf
from data_generator import DataGen


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
        x = ff_layer2 = tf.keras.layers.DepthwiseConv2D(filters, strides, depth_multiplier=1, padding=padding)(input_layer)
    elif conv_type == 'add':
        x = tf.keras.layers.add([input_layer, add_layer])
    else:
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    if relu:
        x = tf.keras.activations.relu(x)
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
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
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
    w = 32
    h = 64
    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)
        concat_list.append(x)
    return tf.keras.layers.concatenate(concat_list)



NO_OF_EPOCHS = 10
BATCH_SIZE = 10



# input image
input_layer = tf.keras.layers.Input(shape=(1024, 2048, 3))
# learning to down sample
lds_layer = conv_layer(input_layer, 'conv', 32, (3, 3), (2, 2)) # size = (1024, 512, 32)
lds_layer = conv_layer(lds_layer, 'ds', 48, (3, 3), (2, 2)) # size = (512, 256, 48)
lds_layer = conv_layer(lds_layer, 'ds', 64, (3, 3), (2, 2)) # size = (256, 128, 64)
# global feature extractor
gfe_layer = bottelneck(lds_layer, 64, (3, 3), 6, 3, (2, 2)) # size = (128, 64, 128)
gfe_layer = bottelneck(gfe_layer, 96, (3, 3), 6, 3, (2, 2)) # size = (128, 64, 128)
gfe_layer = bottelneck(gfe_layer, 128, (3, 3), 6, 3, (1, 1)) # size = (128, 64, 128)
ppm_layer = ppm(gfe_layer, [2,4,6,8])
# Feature Fusion
ff_layer1 = conv_layer(lds_layer, 'conv', 128, (1, 1), (1, 1), relu=False)  # size = (256, 128, 128)
ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(ppm_layer)  # size = (256, 128, 128)
ff_layer2 = conv_layer(ff_layer2, 'dw', 128, (1, 1), (1, 1))  # size = (256, 128, 128)
ff_layer2 = tf.keras.layers.Conv2D(128, (1, 1), (1, 1), padding='same')(ff_layer2)  # size = (256, 128, 128)
ffout_layer = conv_layer(ff_layer2, 'add', 0, (0, 0), (0, 0), add_layer=ff_layer1)  # adding the layers
# # classifier
classifier_layer = conv_layer(ffout_layer, 'ds', 128, (3, 3), (1, 1))  # size = (256, 128, 128)
classifier_layer = conv_layer(classifier_layer, 'ds', 128, (3, 3), (1, 1))  # size = (256, 128, 128)
classifier_layer = conv_layer(classifier_layer, 'conv', 21, (1, 1), (1, 1))  # size = (256, 128, 21)
classifier_layer = tf.keras.layers.Dropout(0.3)(classifier_layer)
classifier_layer = tf.keras.layers.UpSampling2D((8, 8))(classifier_layer)
classifier_layer = tf.keras.activations.softmax(classifier_layer)





fast_scnn_model = tf.keras.Model(inputs=input_layer, outputs=classifier_layer, name='Fast-SCNN')
optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
fast_scnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
fast_scnn_model.summary()


# defining the training generator, validation generator and test generator with batch size 32

# train_gen = data_gen('train', BATCH_SIZE, split=True, data=BATCH_SIZE)
# val_gen = data_gen('val', BATCH_SIZE, split=True, data=BATCH_SIZE)
# test_gen = data_gen('test', BATCH_SIZE, split=True, data=BATCH_SIZE)

train_gen = DataGen('train', batch_size=BATCH_SIZE)
val_gen = DataGen('val', batch_size=BATCH_SIZE)


train_steps = len(train_gen)
valid_steps = len(val_gen)


# setting checkpoints
checkpoint_path = os.path.join(os.getcwd(), "check/cp-{epoch:04d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    save_freq=5)
fast_scnn_model.save_weights(checkpoint_path.format(epoch=0))

# starting to train the model
history = fast_scnn_model.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                                        steps_per_epoch =train_steps,
                                        validation_data=val_gen,
                                        validation_steps=valid_steps,
                                        callbacks=[cp_callback])
