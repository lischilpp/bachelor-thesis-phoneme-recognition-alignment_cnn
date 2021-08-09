import csv

from settings import *


class Phoneme():

    phoneme_list = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr',
        'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh',
        'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g',
        'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
        'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p',
        'pau', 'pcl', 'r', 's', 'sh', 't', 'tcl', 'th',
        'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'q'
    ]

    symbol_to_folded = {
        'ux': 'uw',
        'axr': 'er',
        'ax-h': 'ax',
        'em': 'm',
        'nx': 'n',
        'eng': 'ng',
        'hv': 'hh',
        **dict.fromkeys(['pcl', 'tcl', 'kcl', 'qcl'], 'cl'),
        **dict.fromkeys(['bcl', 'dcl', 'gcl'], 'vcl'),
        **dict.fromkeys(['h#', '#h', 'pau'], 'sil')
    }

    folded_phoneme_list = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch',
        'cl', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er',
        'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k',
        'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
        'sil', 't', 'th', 'uh', 'uw', 'v', 'vcl', 'w',
        'y', 'z', 'zh'
    ]

    symbol_to_folded_group = {
        **dict.fromkeys(['cl', 'vcl', 'epi'], 'sil'),
        'el': 'l',
        'en': 'n',
        'sh': 'zh',
        'ao': 'aa',
        'ih': 'ix',
        'ah': 'ax'
    }

    folded_group_phoneme_list = [
        'aa', 'ae', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh',
        'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ix', 'iy',
        'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r',
        's', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
    ]

    def __init__(self, start, stop, symbol):
        self.start = start
        self.stop = stop
        self.symbol = symbol

    def __str__(self):
        return f'{self.start}-{self.stop}: {symbol}'

    def __repr__(self):
        return self.__str__()

    @classmethod
    def strip_digits(cls, s):
        length = len(s)
        return s[0:length-1] if s[length - 1].isdigit() else s

    @classmethod
    def folded_phoneme_count(cls):
        return len(cls.folded_phoneme_list)

    @classmethod
    def folded_group_phoneme_count(cls):
        return len(cls.folded_group_phoneme_list)

    @classmethod
    def get_phonemes_from_file(cls, path):
        phonemes = []
        with open(path) as pn_file:
            reader = csv.reader(pn_file, delimiter=' ')
            phonemes = []
            for row in reader:
                symbol = cls.strip_digits(row[2])
                if symbol == 'q':
                    continue
                symbol = cls.symbol_to_folded.get(symbol, symbol)
                start = int(row[0])
                stop = int(row[1])
                phoneme = Phoneme(start, stop, symbol)
                phonemes.append(phoneme)

        return phonemes

    @classmethod
    def symbol_to_folded_group_index(cls, symbol):
        return Phoneme.folded_group_phoneme_list.index(Phoneme.symbol_to_folded_group.get(symbol, symbol))