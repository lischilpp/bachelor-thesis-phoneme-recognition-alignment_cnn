from pathlib import Path
from math import floor
import functools
import numpy as np

import tensorflow as tf
from tensorflow import keras


def format_percentage2(n):
    return floor(n * 10000) / 100


model = keras.models.load_model("saved_model")

input_width = 515
input_height = 389
input_size = (input_width, input_height)
batch_size = 32


img_dir = Path('img/test')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    img_dir,
    seed=123,
    image_size=input_size,
    batch_size=batch_size)

class_names = test_ds.class_names
class_count = len(class_names)

predicted_indices = []
actual_indices = []

num_batches = sum([1 for _ in test_ds])

i = 1
for images, labels in test_ds:
    labels_list = list(labels.numpy())
    pred = model.predict(images)
    predicted_list = list(np.argmax(pred, axis=1))
    for predicted, actual in zip(predicted_list, labels_list):
        predicted_indices.append(predicted)
        actual_indices.append(actual)
    print(f'batch {i}/{num_batches}')
    i += 1
classification_counts = [[0 for _ in class_names]
                         for _ in class_names]

correct_predictions = 0
total_predictions = 0
for predicted, actual in zip(actual_indices, predicted_indices):
    classification_counts[actual][predicted] += 1
    if predicted == actual:
        correct_predictions += 1
    total_predictions += 1

accuracies = []
class_index = 0
for counts in classification_counts:
    total = sum(counts)
    actual_score = counts[class_index]
    accuracy = format_percentage2(actual_score / total)

    percentages = []
    for j in range(5):
        highest_score_index = np.argmax(counts)
        highest_score = counts[highest_score_index]
        if highest_score < 0:
            break
        percentage = format_percentage2(highest_score / total)
        percentages.append({"class_index": highest_score_index,
                            "percentage": percentage})
        counts[highest_score_index] = -1

    percentage_str = ""
    j = 0
    for percentage_entry in percentages:
        if j != 0:
            percentage_str += ', '
        class_name = class_names[percentage_entry["class_index"]]
        percentage_str += f'{percentage_entry["percentage"]}% {class_name}'
        j += 1
    accuracies.append({
        "class_name": class_names[class_index],
        "accuracy": accuracy,
        "percentages": percentage_str})

    class_index += 1


def compare(x1, x2):
    return x2["accuracy"] - x1["accuracy"]


accuracies = sorted(
    accuracies, key=functools.cmp_to_key(compare))


print(
    f'accuracy: {format_percentage2(correct_predictions / total_predictions)}%')

f = open('accuracies.csv', "w")

for entry in accuracies:
    f.write(
        f'{entry["accuracy"]}%,{entry["class_name"]},"{entry["percentages"]}"\n')

f.close()
