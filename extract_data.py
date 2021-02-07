import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import csv
from pathlib import Path

timit_path = Path('timit/')
timit_data_path = timit_path / 'data'
data_type = "train"


class Phoneme():
    def __init__(self, start, end, symbol):
        self.start = start
        self.end = end
        self.symbol = symbol

    def __str__(self):
        return f"{self.start}−{self.end}: {self.symbol}"

    def __repr__(self):
        return self.__str__()


def get_phonemes_from_file(path):
    phonemes = []
    with open(pn_path) as pn_file:
        reader = csv.reader(pn_file, delimiter=' ')
        phonemes = [Phoneme(int(row[0]), int(row[1]), row[2])
                    for row in reader][1:-1]
    return phonemes


phoneme_symbols = []
with open('phoneme_symbols.txt') as file:
    phoneme_symbols = file.read().splitlines()

# make directories
Path('img').mkdir(exist_ok=True)
Path('img/train').mkdir(exist_ok=True)
Path('img/test').mkdir(exist_ok=True)
for symbol in phoneme_symbols:
    Path('img/train/'+symbol).mkdir(exist_ok=True)
    Path('img/test/'+symbol).mkdir(exist_ok=True)

# get paths of recordings
recording_paths = []
with open(f'timit/{data_type}_data.csv') as file:
    next(file)
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if row[1] == data_type.upper() and row[10] == 'TRUE':
            path = row[5]
            path_no_ext = path[0:path.index('.')]
            recording_paths.append(path_no_ext)
            # break

i = 0
fig = plt.figure()
# create phoneme images
for path in recording_paths:
    wav_path = timit_data_path / f'{path}.WAV.wav'
    pn_path = timit_data_path / f'{path}.PHN'
    phonemes = get_phonemes_from_file(pn_path)
    spf = wave.open(str(wav_path), 'r')
    if spf.getnchannels() == 2:
        print(f'skipping stereo file: {wav_path}')
        continue
    framerate = spf.getframerate()
    signal = np.frombuffer(spf.readframes(-1), np.int16)
    path_array = path.split('/')
    speaker_id = path_array[-2]
    filename = path_array[-1]
    underscored_path = '_'.join(path_array)

    for phon in phonemes:
        if not phon.symbol in phoneme_symbols:
            continue
        phon_signal = signal[phon.start:phon.end]
        plt.axis("off")
        plt.tick_params(left=False, labelleft=False)  # remove ticks
        plt.box(False)  # remove box
        plt.plot(phon_signal, color="black")
        fig.set_size_inches(2.99, 2.99)
        save_path = f'img/{data_type}/{phon.symbol}/{underscored_path}.png'
        # save_path = f'img/{phon.symbol}{i}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.cla()
        i += 1
plt.close(fig)

print('done')
