""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from . import cmudict

_pad = '<pad>'
_eos = '</s>'
_sos = '<s>'
_punc = '!\'(),-.:~? '

_eng_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

_cmu_characters = [
    'AA', 'AE', 'AH',
    'AO', 'AW', 'AY',
    'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY',
    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH'
]
_cmu_characters = ['@' + s for s in _cmu_characters]


# Export all symbols:
eng_symbols = [_pad, _eos] + list(_eng_characters) + list(_punc) + _arpabet + [_sos]
cmu_symbols = [_pad, _eos] + list(_eng_characters) + list(_punc) + _cmu_characters + [_sos]
