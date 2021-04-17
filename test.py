from pathlib import Path
from math import floor
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from config import *


def format_percentage2(n):
    return floor(n * 10000) / 100


model = keras.models.load_model("saved_model")

input_width = 515
input_height = 389
input_size = (input_width, input_height)
batch_size = 32

input_size = (299, 299)


data_dir = DIRPATH_DATASET/'test'

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=input_size,
    batch_size=batch_size,
)  # color_mode='grayscale')


class_names = test_ds.class_names

#test_ds = test_ds.take(100)

predicted_indices = []
actual_indices = []

i = 1
for images, labels in test_ds:
    labels_list = list(labels.numpy())
    pred = model.predict(images)
    predicted_list = list(np.argmax(pred, axis=1))
    for predicted, actual in zip(predicted_list, labels_list):
        predicted_indices.append(predicted)
        actual_indices.append(actual)
    print(f'batch {i}')
    i += 1

classification_counts = [[0 for _ in class_names]
                         for _ in class_names]

correct_predictions = 0
total_predictions = 0
for actual, predicted in zip(actual_indices, predicted_indices):
    classification_counts[actual][predicted] += 1
    if predicted == actual:
        correct_predictions += 1
    total_predictions += 1

# calculate percentages for all classes


accuracies = []
class_index = 0
for class_counts in classification_counts:
    total_for_class = sum(class_counts)

    accuracy = 0
    percentages = []

    if total_for_class != 0:
        # calculate accuracy for class

        accuracy = class_counts[class_index] / total_for_class

        # calculate top n guessed classes for actual class
        for j in range(5):
            # pick class with highest classification count for the actual class
            class_index_highest_count = np.argmax(class_counts)
            highest_count = class_counts[class_index_highest_count]
            if highest_count < 0:
                break
            acc = 0
            percentage = format_percentage2(highest_count / total_for_class)
            percentages.append({'class_index': class_index_highest_count,
                                'percentage': percentage})
            class_counts[class_index_highest_count] = -1

    # create string for top n guessed classes
    percentage_str = ''
    j = 0
    for percentage_entry in percentages:
        if j != 0:
            percentage_str += ', '
        class_name = class_names[percentage_entry['class_index']]
        percentage_str += f'{percentage_entry["percentage"]}% {class_name}'
        j += 1

    accuracies.append({
        'class_name': class_names[class_index],
        'accuracy': format_percentage2(accuracy),
        'percentages': percentage_str})

    class_index += 1

# sort accuracies in DESC order


def compare(x1, x2):
    return x2["accuracy"] - x1["accuracy"]


accuracies = sorted(
    accuracies, key=functools.cmp_to_key(compare))

total_accuracy = format_percentage2(correct_predictions / total_predictions)

# write result to output file

print(f'accuracy: {total_accuracy}%')

f = open('accuracies.csv', "w")
f.write(f'{total_accuracy}%,Σ,""\n')

for entry in accuracies:
    f.write(
        f'{entry["accuracy"]}%,{entry["class_name"]},"{entry["percentages"]}"\n')

f.close()
