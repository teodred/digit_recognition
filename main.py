import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras


train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')


train_y_full = train.label
train_X_full = train.drop(labels=['label'], axis=1)

train_X_full = train_X_full / 255.
X_test = test / 255.
X_train, X_valid = train_X_full[:-5000], train_X_full[-5000:]
y_train, y_valid = train_y_full[:-5000], train_y_full[-5000:]
X_train = X_train.values.reshape(-1,28,28,1)
X_valid = X_valid.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
y_train = y_train.to_numpy()

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu"),
    keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])


model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

results = model.predict(X_test)

results = np.argmax(results, axis = 1)

results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)

submission.to_csv("data/my_submission.csv", index=False)