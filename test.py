import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import csv


class Phoneme():
    def __init__(self, start, end, symbol):
        self.start = start
        self.end = end
        self.symbol = symbol

    def __str__(self):
        return f"{self.start}−{self.end}: {self.symbol}"

    def __repr__(self):
        return self.__str__()


wav_path = 'timit/data/TRAIN/DR1/FSJK1/SA1.WAV.wav'
pn_path = 'timit/data/TRAIN/DR1/FSJK1/SA1.PHN'

spf = wave.open(wav_path, 'r')
signal = np.frombuffer(spf.readframes(-1), np.int16)
signal_length = len(signal)

# If Stereo
if spf.getnchannels() == 2:
    print(f'skipping stereo file: {wav_path}')
    sys.exit(0)

framerate = spf.getframerate()
duration = len(signal) / framerate


def get_signal_at(time):
    index = (int)(time / duration * signal_length)
    return signal[index]


phonemes = []
with open(pn_path) as pn_file:
    reader = csv.reader(pn_file, delimiter=' ')
    phonemes = [Phoneme(int(row[0]), int(row[1]), row[2])
                for row in reader][1:-1]


# phoneme = signal[2260:4149]

# Time = np.linspace(0, len(phoneme) / framerate, num=len(phoneme))

# plt.figure(1)
# plt.title("Signal Wave...")
# plt.plot(Time, phoneme)
# plt.show()
