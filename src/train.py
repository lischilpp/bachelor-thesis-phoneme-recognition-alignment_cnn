import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_hub as hub
from sklearn.utils import class_weight
import numpy as np
from pathlib import Path
import os
from settings import *

if len(tf.config.list_physical_devices('GPU')) >= 1:
    print("GPU is available")
else:
    print("GPU not available")


model_url, pixels = (
    "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4", 299)

input_size = (pixels, pixels)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.05,
    subset="training",
    seed=123,
    image_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.05,
    subset="validation",
    seed=123,
    image_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
)

num_classes = len(train_ds.class_names)

do_fine_tuning = True
print("Building model with", model_url)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_size + (3,)),
    hub.KerasLayer(model_url, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(len(train_ds.class_names),
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model.build([None, *input_size, 3])
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

earlystop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=5)

checkpoint_dir = Path('./checkpoint')
checkpoint_path = checkpoint_dir / 'cp.ckpt'

if checkpoint_dir.exists():
    model.load_weights(checkpoint_path)
    print('loaded from checkpoint')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


class_weights = None

class_indices = []
i = 0
for class_name in sorted(os.listdir(TRAIN_PATH)):
    image_count = sum([1 for _ in os.listdir(
        TRAIN_PATH / class_name)])
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
    class_weight=class_weights,
    callbacks=[cp_callback, tensorboard_callback]
).history

model.save("saved_model")
