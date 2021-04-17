import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_hub as hub
from sklearn.utils import class_weight
import numpy as np
from pathlib import Path
import os
from config import *

if len(tf.config.list_physical_devices('GPU')) >= 1:
    print("GPU is available")
else:
    print("GPU not available")

data_dir = DIRPATH_DATASET/'train'

input_width = 299
input_height = 299
input_size = (input_width, input_height)
batch_size = 16

model_url, pixels = (
    "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4", 299)

input_size = (pixels, pixels)

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

num_classes = len(train_ds.class_names)

do_fine_tuning = True
print("Building model with", model_url)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_size + (3,)),
    hub.KerasLayer(model_url, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(len(train_ds.class_names),
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

# model = tf.keras.Sequential([
#     layers.experimental.preprocessing.Rescaling(
#         1./255, input_shape=(*input_size, 1)),
#     layers.Conv2D(16, 1, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 1, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 1, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(len(train_ds.class_names))
# ])


model.build([None, *input_size, 3])
model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# earlystop_callback = EarlyStopping(
#     monitor='val_accuracy',
#     min_delta=0.0001,
#     patience=3)

checkpoint_dir = Path('./checkpoint')
checkpoint_path = checkpoint_dir / 'cp.ckpt'

if checkpoint_dir.exists():
    model.load_weights(checkpoint_path)
    print('loaded from checkpoint')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)

class_weights = None

class_indices = []
i = 0
for class_name in sorted(os.listdir(data_dir)):
    image_count = len([f for f in os.listdir(
        data_dir / class_name)])
    class_indices += [i] * image_count
    i += 1

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_indices),
    y=class_indices)

class_weights = {i: class_weights[i] for i in range(len(class_weights))}

hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=300,
    callbacks=[cp_callback]
).history

model.save("saved_model")
