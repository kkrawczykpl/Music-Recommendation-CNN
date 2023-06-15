from utils.constants import MODEL_DIR_NAME
from utils.file_handler import create_if_not_exists
from utils.spectogram import create_spectograms, load_dataset, split_spectogram
from keras import initializers, optimizers
import pandas as pd
from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Flatten, Dropout, Dense
from keras.models import Sequential

create_if_not_exists(MODEL_DIR_NAME)

create_spectograms()
split_spectogram()

train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(datasetSize=0.7)


train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

train_x = train_x / 255.
test_x = test_x / 255.

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=[7,7], kernel_initializer = initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
# Dim = (122x122x64)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (61x61x64)
model.add(Conv2D(filters=128, kernel_size=[7,7], strides=2, kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (28x28x128)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (14x14x128)
model.add(Conv2D(filters=256, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (12x12x256)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (6x6x256)
model.add(Conv2D(filters=512, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (4x4x512)
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (2x2x512)
model.add(BatchNormalization())
model.add(Flatten())
# Dim = (2048)
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(1024, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (1024)
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (256)
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (64)
model.add(Dense(32, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (32)
model.add(Dense(n_classes, activation="softmax", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (8)
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

pd.DataFrame(model.fit(train_x, train_y, epochs=10, verbose=1, validation_split=0.1).history).to_csv(f'{MODEL_DIR_NAME}/training_history.csv')

score = model.evaluate(test_x, test_y, verbose=1)

print(score)

model.save(f'{MODEL_DIR_NAME}/Model.h5')