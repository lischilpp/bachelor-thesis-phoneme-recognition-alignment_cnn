from pathlib import Path
from math import floor
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from settings import *
from phonemes import Phoneme


def format_percentage2(n):
    return floor(n * 10000) / 100

def show_confusion_matrix(confmat):
    plt.figure(figsize=(15,10))

    class_names = Phoneme.folded_group_phoneme_list
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_predictions_and_labels():
    model = keras.models.load_model(SAVED_MODEL_PATH)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PATH,
        seed=123,
        image_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
    )
    #test_ds = test_ds.take(100)

    class_names = test_ds.class_names

    all_predictions_list = []
    all_labels_list = []
    i = 0
    for images, labels in test_ds:
        labels_list = list(labels.numpy())
        pred = model.predict(images)
        predicted_list = list(np.argmax(pred, axis=1))
        for predicted, actual in zip(predicted_list, labels_list):
            predicted = Phoneme.symbol_to_folded_group_index(predicted)
            actual    = Phoneme.symbol_to_folded_group_index(actual)
            all_predictions_list.append(predicted)
            all_labels_list.append(actual)
            
        print(f'batch {i}')
        i += 1
    
    return all_predictions_list, all_labels_list


predictions, labels = get_predictions_and_labels()

correct_predictions_count = 0
total_predictions_count = 0
for p, l in zip(predictions, labels):
    if p == l:
        correct_predictions_count += 1
    total_predictions_count += 1

accuracy = format_percentage2(correct_predictions_count / total_predictions_count)

print(f'Accuracy: {accuracy}%')
confmat = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=Phoneme.folded_group_phoneme_count())
show_confusion_matrix(confmat)