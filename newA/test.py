import cv2
import numpy as np
from keras_tuner.src.backend.io import tf
from matplotlib import pyplot as plt
import pandas as pd


from newA.const import train_path, saved_model_path, size_x, size_y, train, max_length


def data_generator(df=train, batch_size=16):

    while True:  # if you add a while loop,the generator will give product once not anymore
        images = np.zeros((batch_size, size_x, size_y, 3))  # (380,676,3) is all images' size
        bbox_coords = np.zeros((batch_size, max_length))

        for i in range(batch_size):
            rand_index = np.random.randint(0, train.shape[0])
            row = df.loc[rand_index, :]
            image = cv2.imread(str(train_path / row.image)) / 255.
            images[i] = image
            bbox_coords[i] = np.array(row.coordinates)

        yield {'image': images}, {'coordinates': bbox_coords}



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
        print(pred_bbox)
        display_image(img, pred_coords=pred_bbox, normalize=True)
    plt.show()

def test_f(model):
    print("Start")
    for i in range(10):
        test(model)

loaded_model = tf.keras.models.load_model(saved_model_path)
test_f(loaded_model)
