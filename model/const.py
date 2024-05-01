import pandas as pd
from pathlib import Path

size_x = 380
size_y = 676
train_path = Path(r"C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\dataset\data\training_images")
test_path = Path(r"C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\dataset\data\testing_images")
saved_model_path = r'C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\newA\saved_model.h5'

train = pd.read_csv(
    r"C:\Users\PC\PycharmProjects\AMA_many_car_object_detection\dataset\data\train_solution_bounding_boxes.csv")
train = train.groupby("image")[["xmin", "ymin", "xmax", "ymax"]].apply(
    lambda x: x.values.astype(int).flatten().tolist()).reset_index()
train.rename(columns={0: "coordinates"}, inplace=True)
max_length = train["coordinates"].apply(len).max()
train["coordinates"] = train["coordinates"].aggregate(
    lambda x: x + [0] * (max_length - len(x)))  # equalize the coordinates with empty boxes for no needed ones

