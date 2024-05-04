import cv2
import numpy as np
from keras_tuner.src.backend.io import tf
from matplotlib import pyplot as plt
import pandas as pd

from const import size_x, size_y, max_length

def data_generator(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size_y, size_x))  # Змінюємо розмір до очікуваного
    image = image / 255.  # Нормалізуємо значення пікселів до [0, 1]
    images = np.expand_dims(image, axis=0)
    bbox_coords = np.zeros((1, max_length))  # Assuming single image, so batch size is 1
    return {'image': images}, {'coordinates': bbox_coords}


def display_image(img, bbox_coords=[], pred_coords=[], normalize=False):
    """
    Function: Receives an image and display it with boundary box
    parameters:
        bbox_coords : If True,shows boundary box(es) of the specific image(s) with green color line
        pred_coords : If True,shows the predicted boundary boxes with red color line
        normalize   : Rescales the channels if normalize is True
    """

    if normalize:
        img *= 255.
        img = img.astype(np.uint8)

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

def test(model, image_path):
    plt.figure(figsize=(5, 5))
    example, label = data_generator(image_path)
    X = example['image']
    y = label['coordinates']

    img = X[0]
    gt_coords = y[0]
    pred_bbox = model.predict(X)[0]
    print(pred_bbox)
    display_image(img, pred_coords=pred_bbox, normalize=True)
    plt.show()

def test_f(model, image_path):
    print("Testing on image:", image_path)
    test(model, image_path)

saved_model_path = r"C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\newA\saved_model.h5"
image_path = r"C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\dataset\photo_2024-05-04_18-15-01.jpg"
loaded_model = tf.keras.models.load_model(saved_model_path)
test_f(loaded_model, image_path)
