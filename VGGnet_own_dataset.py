import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import random
import math
from PIL import Image
import os
import numpy as np
##
batch_size = 32  # for train & val. modify according to your GPU memory and data
num_classes = 2  # total classes to train
base_lr = 1e-3  # learning rate
num_epoch = 10  # go through your training & val data epoch times
img_height = img_width = 224  # picture size (imagenet challenge)
channel = 3  # RGB=3, grayscale=1
seed = random.randint(0, 100)
checkpoint_path = './checkpoints/weights.{epoch:02d}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5'  # path for saving checkpoints
##
data_gen_args = dict(rescale=1./255,
                     rotation_range=5,
                     shear_range=0.2,
                     zoom_range=0.2,
                     vertical_flip=True,
                     horizontal_flip=True)

# train data generator
train_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# TODO : Add your own full path training data folder (training_path)
train_generator = train_datagen.flow_from_directory(
        r'your path of training data folder',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)
##
# val data generator
# TODO : Add your own full path validation data folder (validation_path)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        r'your path of validation data folder',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)
##
# TODO : You can change your model if you want.
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(img_width, img_height, channel), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())
##
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate

callbacks = [keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True),
             keras.callbacks.LearningRateScheduler(step_decay)]
##
optim = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
##
history = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=STEP_SIZE_TRAIN,
                                     validation_data=validation_generator,
                                     validation_steps=STEP_SIZE_VALID,
                                     epochs=num_epoch,
                                     callbacks=callbacks,
                                     initial_epoch=0,
                                     verbose=2,
                                     workers=1)
##plot model loss & save
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('classifier_loss.png')
## plot model accuracy & save
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('classifier_acc.png')
##

# TODO : Add your own full path testing data folder (testing_path)
test_path = r'test folder'
for file in os.listdir(test_path):
    im = Image.open(test_path + file)
    im = im.resize((224,224))
    x = np.asarray(im)
    x = x / 255.
    batch_x = np.asarray([x])
    result = model.predict(batch_x)
    result_index = np.argmax(result[0], axis=0)
    if result_index == 1:
        print(file+' is dog')
    else:
        print(file + ' is cat')
