import os
import shutil
import random
import keras
from keras import layers
from keras import models
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# mkdir a operate folder
dataset = '/Users/wongshnyau/Downloads/dataset'
operation = '/Users/wongshnyau/Downloads/operation'
os.mkdir(operation)

# mkdir a train, validation, and test folder
train_dir = os.path.join(operation, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(operation, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(operation, 'test')
os.mkdir(test_dir)

# copy pictures to above folder
for i in range(1, 8):
    # mkdir of 7 class (1,2,3,4,5,6,7) 不確定g是什麼
    train_dirr = os.path.join(train_dir, 'train_%s' % (str(i)))
    os.mkdir(train_dirr)
    validation_dirr = os.path.join(validation_dir, 'validation_%s' % (str(i)))
    os.mkdir(validation_dirr)
    test_dirr = os.path.join(test_dir, 'test_%s' % (str(i)))
    os.mkdir(test_dirr)

    # random copy pictures to those directory by 2/1/1
    original_dir = os.listdir(os.path.join(dataset, str(i)))
    sample = random.sample(original_dir, len(original_dir) // 2)
    sample2 = list(set(original_dir) - set(sample))
    sample3 = random.sample(sample2, len(sample2) // 2)
    sample4 = list(set(sample2) - set(sample3))

    # copy
    for pic_name in sample:
        A = os.path.join(dataset, str(i), pic_name)
        B = os.path.join(train_dirr, pic_name)
        shutil.copyfile(A, B)

    for pic_name in sample3:
        A = os.path.join(dataset, str(i), pic_name)
        B = os.path.join(validation_dirr, pic_name)
        shutil.copyfile(A, B)

    for pic_name in sample4:
        A = os.path.join(dataset, str(i), pic_name)
        B = os.path.join(test_dirr, pic_name)
        shutil.copyfile(A, B)


# set model
model = models.Sequential()
# convolution layer is a feature detector, help model to get some profile of pictures
# relu as active function can highlight features
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(30, 30, 3)))
# pooling layer can downsampling, in this case, it can minimize influence of difference of character location
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# add drop out layer to avoid overvitting
model.add(Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# rescales all images by 1/255

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    # resizes all images to 30*30
    target_size=(30, 30),
    batch_size=10,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(30, 30),
    batch_size=10,
    class_mode='categorical')


history = model.fit_generator(
    train_generator,
    # train size = 100 * 12
    steps_per_epoch=100,
    epochs=12,
    validation_data=validation_generator,
    validation_steps=50)

# plt accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(28, 28),
    batch_size=1,
    class_mode='categorical')


loss, acc = model.evaluate_generator(test_generator, steps=len(test_generator.filenames))
# loss = 0.000188903295227477
# acc = 1
