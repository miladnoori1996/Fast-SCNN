import os
import tensorflow as tf
import numpy as np
from model import conv_layer, bottelneck, ppm
from new_datagen import DataGen


NO_OF_EPOCHS = 1000
BATCH_SIZE = 2
SAMPLES = 2975
VAL_SAMPLES = 500



# input image
input_layer = tf.keras.layers.Input(shape=(256, 512, 3))
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
ff_layer1 = conv_layer(lds_layer, 'conv', 128, (1, 1), (1, 1), relu=False)  # size = (256, 128, 12 = tf.keras.layers.UpSampling2D((4, 4))(ppm_layer)  # size = (256, 128, 128)
ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(ppm_layer)  # size = (256, 128, 128)
ff_layer2 = tf.keras.layers.DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')(ff_layer2)
ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
ff_layer2 = tf.keras.layers.Conv2D((128), (1, 1), (1, 1), padding='same')(ff_layer2)  # size = (256, 128, 128)
ffout_layer = conv_layer(ff_layer2, 'add', 0, (0, 0), (0, 0), add_layer=ff_layer1)  # adding the layers
# # classifier
classifier_layer = conv_layer(ffout_layer, 'ds', 128, (3, 3), (1, 1))  # size = (256, 128, 128)
classifier_layer = conv_layer(classifier_layer, 'ds', 128, (3, 3), (1, 1))  # size = (256, 128, 128)
classifier_layer = conv_layer(classifier_layer, 'conv', 21, (1, 1), (1, 1))  # size = (256, 128, 1)
classifier_layer = tf.keras.layers.Dropout(0.3)(classifier_layer)
classifier_layer = tf.keras.layers.UpSampling2D((8, 8))(classifier_layer)
classifier_layer = tf.keras.activations.softmax(classifier_layer)

# compiling the model
fast_scnn_model = tf.keras.Model(inputs=input_layer, outputs=classifier_layer, name='Fast-SCNN')
optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
fast_scnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
fast_scnn_model.summary()
# categorical_crossentropy
# sparse_categorical_crossentropy



train_gen = DataGen('train', batch_size=BATCH_SIZE, image_height=256, image_width=512)
val_gen = DataGen('val', batch_size=BATCH_SIZE, image_height=256, image_width=512)
# test_gen = DataGen('test', batch_size=BATCH_SIZE)


# setting checkpoints
checkpoint_path = os.path.join(os.getcwd(), "check/cp-{epoch:04d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    save_freq=5)
fast_scnn_model.save_weights(checkpoint_path.format(epoch=0))

# starting to train the model
history = fast_scnn_model.fit_generator(train_gen, 
										epochs=NO_OF_EPOCHS,
                                        steps_per_epoch =SAMPLES//BATCH_SIZE,
                                        validation_data=val_gen,
                                        validation_steps=VAL_SAMPLES//BATCH_SIZE,
                                        verbose=1,
                                        callbacks=[cp_callback])







