import os
import tensorflow as tf
from data_generator import DataGen
from model import conv_layer, bottelneck, ppm
import cv2
# from data_generator2 import seg_gen
# from tensorflow.python import debug as tf_debug



NO_OF_EPOCHS = 1000
BATCH_SIZE = 32
samples = 2975
steps = samples//BATCH_SIZE


# input image
input_layer = tf.keras.layers.Input(shape=(1024, 2048, 3))
# learning to down sample
lds_layer = conv_layer(input_layer, 'conv', 32, (3, 3), (2, 2)) # size = (512, 1024, 32)
lds_layer = conv_layer(lds_layer, 'ds', 48, (3, 3), (2, 2)) # size = (256, 512, 48)
lds_layer = conv_layer(lds_layer, 'ds', 64, (3, 3), (2, 2)) # size = (128, 256, 64)
# # global feature extractor
gfe_layer = bottelneck(lds_layer, 64, (3, 3), 6, 3, (2, 2)) # size = (64, 128, 128)
gfe_layer = bottelneck(gfe_layer, 96, (3, 3), 6, 3, (2, 2)) # size = (64, 128, 128)
gfe_layer = bottelneck(gfe_layer, 128, (3, 3), 6, 3, (1, 1)) # size = (64, 128, 128)
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

# compiling the model
fast_scnn_model = tf.keras.Model(inputs=input_layer, outputs=classifier_layer, name='Fast-SCNN')
optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
fast_scnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
fast_scnn_model.summary()

# defining the training generator, validation generator and test generator with batch size 32

# train_gen = data_gen('train', BATCH_SIZE, split=True, data=BATCH_SIZE)
# val_gen = data_gen('val', BATCH_SIZE, split=True, data=BATCH_SIZE)
# test_gen = data_gen('test', BATCH_SIZE, split=True, data=BATCH_SIZE)

train_gen = DataGen('train', batch_size=BATCH_SIZE, image_width=1024, image_height=2048)
val_gen = DataGen('val', batch_size=BATCH_SIZE, image_width=1024, image_height=2048)
test_gen = DataGen('test', batch_size=BATCH_SIZE, image_width=1024, image_height=2048)

train_steps = len(train_gen)
valid_steps = len(val_gen)

# x, y = train_gen.__getitem__(0)

# print(x.shape, y.shape)
# cv2.imshow("new", y[0][:,:,0])
# cv2.waitKey(0)



# USING DATA_GENERATOR2
# train_gen = seg_gen('train', batch_size=BATCH_SIZE, img_height=256, img_width=512)




# setting checkpoints
checkpoint_path = os.path.join(os.getcwd(), "check/cp-{epoch:04d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    save_freq=5)
fast_scnn_model.save_weights(checkpoint_path.format(epoch=0))

# starting to train the model
# history =fast_scnn_model.fit_generator(train_gen, verbose=1, steps_per_epoch=steps, epochs=8, callbacks=[cp_callback])

history = fast_scnn_model.fit_generator(train_gen, epochs=NO_OF_EPOCHS,
                                        steps_per_epoch =train_steps,
                                        validation_data=val_gen,
                                        validation_steps=valid_steps,
                                        callbacks=[cp_callback])









