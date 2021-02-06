import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path
import os

if len(tf.config.list_physical_devices('GPU')) >= 1:
    print("GPU is available")
else:
    print("GPU not available")

data_dir = Path('img/')

input_width = 515
input_height = 389
input_size = (input_width, input_height)
batch_size = 32

# (train_images, train_labels), (test_images,
#                                test_labels) = datasets.cifar10.load_data()

# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=input_size,
    batch_size=batch_size,
)  # color_mode='grayscale')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=input_size,
    batch_size=batch_size,
)  # color_mode='grayscale')


model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(input_height, input_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names))
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

earlystop_callback = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.0001)

checkpoint_path = 'checkpoint/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)

hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[earlystop_callback, cp_callback]
).history

model.save("saved_model")
