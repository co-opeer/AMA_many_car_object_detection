# Data Operations
import numpy as np
from keras_tuner.src.backend import keras

# Visualization/Image Processing
import cv2
import matplotlib.pyplot as plt

# Machine Learning
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, MaxPool2D, Dense

import glob

from newA.const import saved_model_path, train, max_length, size_x, size_y

train_images_paths = glob.glob(
    r"C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\dataset\data\training_images\*.jpg")
test_images_paths = glob.glob(
    r"C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\dataset\data\testing_images\*.jpg")
print(f"Train images data size is {len(train_images_paths)} besides test images data size is {len(test_images_paths)}")


len(train)

# Checking the length of coordinates
train["coordinates"].apply(len).min()  # So far all we have done is nice.


# We define a function for using either now or later
def display_image(img, bbox_coords=[], pred_coords=[], normalize=False):
    """
    Function: Receives an image and display it with boundary box
    parameters:
        bbox_coords : If True,shows boundary box(es) of the specific image(s) with green color line
        pred_coords : If True,shows the predicted boundary boxes with red color line
        normalize   : Rescales the channels if normalize is True
    """

    if normalize:
        img = img / 255

    while len(pred_coords) > 0:
        xmin, ymin, xmax, ymax = pred_coords[:4]
        pred_coords = pred_coords[4:]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), thickness=2)  # Red channel

    while len(bbox_coords) > 0:
        xmin, ymin, xmax, ymax = bbox_coords[:4]
        bbox_coords = bbox_coords[4:]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), thickness=2)  # Green channel

    plt.imshow(img)
    plt.axis("off")


# Display an image with boxes
train["image_path"] = r"C:/Users/PC/PycharmProjects/AMA_many_car_object_detection/dataset/data/training_images/" + \
                      train["image"]
display_image(cv2.imread(train["image_path"][0]))

# Now let's see the same image with boxes
display_image(cv2.imread(train["image_path"][0]), bbox_coords=train["coordinates"][0])

# The one  within multiple cars and boxes
display_image(cv2.imread(train["image_path"][48]), bbox_coords=train["coordinates"][48])


# Here we put python expression yield insted of return because in generator it is used yield for many reasons.
# To learn about generators in python please visite "https://www.programiz.com/python-programming/generator"
def data_generator(df=train, batch_size=16):
    """
        Function: returns images in batch_size with the bbox coordinates they have
    """
    while True:  # if you add a while loop,the generator will give product once not anymore
        images = np.zeros((batch_size, size_x, size_y, 3))  # (380,676,3) is all images' size
        bbox_coords = np.zeros((batch_size, max_length))

        for i in range(batch_size):
            rand_index = np.random.randint(0, train.shape[0])
            row = df.loc[rand_index, :]
            images[i] = cv2.imread(row.image_path) / 255.
            bbox_coords[i] = np.array(row.coordinates)

        yield {'image': images}, {'coordinates': bbox_coords}


# Test our generator
img, label = next(data_generator(batch_size=1))
img = img["image"][0]
bbox_coords = label['coordinates'][0]
display_image(img, bbox_coords=bbox_coords)

shape = (size_x, size_y, 3)
inputs = Input(shape=shape, name="image")  # image data come from data_generator
x = inputs
n_filters = 0
for i in range(5):  # Convolutional Blocks
    n_filters = 2 ** (i + 2)
    x = Conv2D(n_filters, 3, activation="relu", padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2, padding='same')(x)

x = Flatten()(x)
x = Dense(256, "relu")(x)
x = Dense(32, "relu")(x)
output = Dense(28, activation='relu', name='coordinates')(x)  # coordinates data come from data_generator

model = tf.keras.models.Model(inputs, output)
model.summary()

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy']

)


# Some functions to test the model. These will be called every epoch to display the current performance of the model
def test(model):
    # Show three images for testing
    plt.figure(figsize=(15, 7))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        example, label = next(data_generator(batch_size=1))  # bbox_coords=[],pred_coords=
        X = example['image']
        y = label['coordinates']

        img = X[0]
        gt_coords = y[0]
        pred_bbox = model.predict(X)[0]
        display_image(img, bbox_coords=gt_coords, pred_coords=pred_bbox)
    plt.show()


class TestImages(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test(self.model)


# Testing everything before process is not a wasting time. Test it now
test(model)

# train the model
with tf.device('/GPU:0'):
    _ = model.fit(data_generator(), epochs=9, steps_per_epoch=500, callbacks=[TestImages()])
    keras.saving.save_model(model, saved_model_path)
