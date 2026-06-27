
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from setup import path

test_data = path + "/emnist-balanced-test.csv"

training_data = path + "/emnist-balanced-train.csv"

map_to_asci = path + "/emnist-balanced-mapping.txt"

df_test = pd.read_csv(
    test_data
)

df_tr = pd.read_csv(
    training_data
)

def process(df): # returns labels, x train
    labels = df.iloc[:,0]
    #print(labels.shape)
    #print(labels.describe())

    images = df.iloc[:, 1:]

    #print(images.shape)
    #print(images.describe())

    #Testing first image in first row. 
    image = images.values
    reshaped_image = image.reshape(-1, 28, 28, 1)
    test_image = np.transpose(reshaped_image.squeeze(), axes=(0, 2, 1)) # remove 1d color channel and flip the original image 90 degrees so width (1) axis becomes height axis (2), and height axis (2) becomes width axis (1)
    gray_scale = test_image.astype("float32")/255

    
    return labels, np.expand_dims(gray_scale, axis=-1)


if __name__ == "__main__":
    (y_train, x_train) = process(df_tr)



    (y_test, x_test) = process(df_test)


    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    # created training, test, validation dataset.

    model = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.Conv2D(filters=20, kernel_size=3, activation="relu", padding="same", input_shape=(28,28,1)),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(filters=40, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(filters=60, kernel_size=3, activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(47, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        min_delta = 0.001,
        patience = 20,
        restore_best_weights=True
    )
    history = model.fit(
        x_train, y_train,
        validation_data = (x_val, y_val),
        batch_size = 100,
        epochs = 10,
        callbacks = [early_stopping],
    )

    model.save("emnist_cnn.keras")

    plt.plot(history.history["loss"], label = "loss")
    plt.plot(history.history["val_loss"], label = "val loss")
    plt.plot(history.history["accuracy"], label = "Accuracy")
    plt.plot(history.history["val_accuracy"], label = "val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("")
    plt.legend()

    plt.show()