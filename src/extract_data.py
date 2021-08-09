import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import wave
import sys
import csv
from pathlib import Path, PurePath
from scipy import signal as sksignal
from scipy.io import wavfile
from phonemes import Phoneme
from settings import *

core_test_set_speakers = [
    'MDAB0', 'MTAS1', 'MJMP0', 'MLLL0',
    'MBPM0', 'MCMJ0', 'MGRT0', 'MJLN0',
    'MWBT0', 'MWEW0', 'MLNT0', 'MTLS0',
    'MKLT0', 'MJDH0', 'MNJM0', 'MPAM0',
    'FELC0', 'FPAS0', 'FPKT0', 'FJLM0',
    'FNLP0', 'FMGD0', 'FDHC0', 'FMLD0'
]

def make_directories():
    DATA_PATH.mkdir(exist_ok=True)
    TRAIN_PATH.mkdir(exist_ok=True)
    TEST_PATH.mkdir(exist_ok=True)
    for symbol in Phoneme.folded_phoneme_list:
        Path(TRAIN_PATH / symbol).mkdir(exist_ok=True)
        Path(TEST_PATH / symbol).mkdir(exist_ok=True)

def get_recording_paths(test=False):
    recording_paths = []
    train_test_str = "TEST" if test else "TRAIN"
    for path in (TIMIT_PATH / train_test_str).rglob('*.WAV.wav'):
        path_entries = PurePath(path).parts
        speaker_id = path_entries[5]
        filename = path_entries[6]
        if filename.startswith('SA') or (test and not speaker_id in core_test_set_speakers):
            continue
        recording_paths.append(str(path.relative_to(TIMIT_PATH))[:-8])
    return recording_paths

def create_spectrograms_from_audio_paths(audio_paths, save_dir):
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    # create phoneme images
    for path in audio_paths:
        wav_path = TIMIT_PATH / f'{path}.WAV.wav'
        pn_path = TIMIT_PATH / f'{path}.PHN'
        phonemes = Phoneme.get_phonemes_from_file(pn_path)
        spf = wave.open(str(wav_path), 'r')
        if spf.getnchannels() == 2:
            print(f'skipping stereo file: {wav_path}')
            continue
        signal, sampling_rate = librosa.load(wav_path)
        path_array = path.split('/')
        speaker_id = path_array[-2]
        filename = path_array[-1]
        underscored_path = '_'.join(path_array)

        for phon in phonemes:
            if phon.symbol == 'q':
                continue
            phon_signal = signal[phon.start:phon.stop]
            plt.axis("off")
            plt.tick_params(left=False, labelleft=False)
            plt.box(False)
            mel_spect = librosa.feature.melspectrogram(y=phon_signal, sr=sampling_rate, n_fft=128, n_mels=40, hop_length=32)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
            librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');        
            fig.set_size_inches(2.99, 2.99)
            save_path = save_dir / f'{phon.symbol}/{underscored_path}.png'
            fig.savefig(save_path, dpi=100)
            plt.cla()
    plt.close(fig)


make_directories()
print('reading training paths...')
train_paths = get_recording_paths(test=False)
print(len(train_paths))
exit()
print('creating training images...')
create_spectrograms_from_audio_paths(train_paths, TRAIN_PATH)
print('reading test paths...')
test_paths = get_recording_paths(test=True)
print('creating test images...')
create_spectrograms_from_audio_paths(test_paths, TEST_PATH)  
print('done')
