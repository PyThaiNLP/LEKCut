# -*- coding: utf-8 -*-
"""
Deepcut: A Thai word tokenization library using Deep Neural Network

Rakpong Kittinaradorn, Titipat Achakulvisut, Korakot Chaovavanich, Kittinan Srithaworn,
Pattarawat Chormai, Chanwit Kaewkasi, Tulakan Ruangrong, Krichkorn Oparad.
(2019, September 23). DeepCut: A Thai word tokenization library using Deep Neural Network. Zenodo. http://doi.org/10.5281/zenodo.3457707

License: MIT License (For Model and Code that come from Deepcut's GitHub)
GitHub: https://github.com/rkcosmos/deepcut
"""
from typing import List

from lekcut.model import get_path
import onnxruntime as ort
import numpy as np

CHAR_TYPE = {
    u'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    u'ฅฉผฟฌหฮ': 'n',
    u'ะาำิีืึุู': 'v',  # า ะ ำ ิ ี ึ ื ั ู ุ
    u'เแโใไ': 'w',
    u'่้๊๋': 't', # วรรณยุกต์ ่ ้ ๊ ๋
    u'์ๆฯ.': 's', # ์  ๆ ฯ .
    u'0123456789๑๒๓๔๕๖๗๘๙': 'd',
    u'"': 'q',
    u"‘": 'q',
    u"’": 'q',
    u"'": 'q',
    u' ': 'p',
    u'abcdefghijklmnopqrstuvwxyz': 's_e',
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}

CHAR_TYPE_FLATTEN = {}
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v

# create map of dictionary to character
CHARS = [
    u'\n', u' ', u'!', u'"', u'#', u'$', u'%', u'&', "'", u'(', u')', u'*', u'+',
    u',', u'-', u'.', u'/', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8',
    u'9', u':', u';', u'<', u'=', u'>', u'?', u'@', u'A', u'B', u'C', u'D', u'E',
    u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
    u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z', u'[', u'\\', u']', u'^', u'_',
    u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
    u'n', u'o', u'other', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    u'z', u'}', u'~', u'ก', u'ข', u'ฃ', u'ค', u'ฅ', u'ฆ', u'ง', u'จ', u'ฉ', u'ช',
    u'ซ', u'ฌ', u'ญ', u'ฎ', u'ฏ', u'ฐ', u'ฑ', u'ฒ', u'ณ', u'ด', u'ต', u'ถ', u'ท',
    u'ธ', u'น', u'บ', u'ป', u'ผ', u'ฝ', u'พ', u'ฟ', u'ภ', u'ม', u'ย', u'ร', u'ฤ',
    u'ล', u'ว', u'ศ', u'ษ', u'ส', u'ห', u'ฬ', u'อ', u'ฮ', u'ฯ', u'ะ', u'ั', u'า',
    u'ำ', u'ิ', u'ี', u'ึ', u'ื', u'ุ', u'ู', u'ฺ', u'เ', u'แ', u'โ', u'ใ', u'ไ',
    u'ๅ', u'ๆ', u'็', u'่', u'้', u'๊', u'๋', u'์', u'ํ', u'๐', u'๑', u'๒', u'๓',
    u'๔', u'๕', u'๖', u'๗', u'๘', u'๙', u'‘', u'’', u'\ufeff'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}

CHAR_TYPES = [
    'b_e', 'c', 'd', 'n', 'o',
    'p', 'q', 's', 's_e', 't',
    'v', 'w'
]
CHAR_TYPES_MAP = {v: k for k, v in enumerate(CHAR_TYPES)}


def create_feature_array(text, n_pad=21):
    """
    Create feature array of character and surrounding characters
    """
    n = len(text)
    n_pad_2 = int((n_pad - 1)/2)
    text_pad = [' '] * n_pad_2  + [t for t in text] + [' '] * n_pad_2
    x_char, x_type = [], []
    for i in range(n_pad_2, n_pad_2 + n):
        char_list = text_pad[i + 1: i + n_pad_2 + 1] + \
                    list(reversed(text_pad[i - n_pad_2: i])) + \
                    [text_pad[i]]
        char_map = [CHARS_MAP.get(c, 80) for c in char_list]
        char_type = [CHAR_TYPES_MAP.get(CHAR_TYPE_FLATTEN.get(c, 'o'), 4)
                     for c in char_list]
        x_char.append(char_map)
        x_type.append(char_type)
    x_char = np.array(x_char).astype(float)
    x_type = np.array(x_type).astype(float)
    return x_char, x_type


def create_n_gram_df(df, n_pad):
    """
    Given input dataframe, create feature dataframe of shifted characters
    """
    n_pad_2 = int((n_pad - 1)/2)
    for i in range(n_pad_2):
        df['char-{}'.format(i+1)] = df['char'].shift(i + 1)
        df['type-{}'.format(i+1)] = df['type'].shift(i + 1)
        df['char{}'.format(i+1)] = df['char'].shift(-i - 1)
        df['type{}'.format(i+1)] = df['type'].shift(-i - 1)
    return df[n_pad_2: -n_pad_2]

_TOKENIZER = None
def tokenize(text: str, path: str="default") -> List[str]:
    global _TOKENIZER
    if path=="default":
        path = get_path("deepcut.onnx")
    if _TOKENIZER == None:
        _TOKENIZER = Tokenizer(path=path)
    elif _TOKENIZER.path != path:
        _TOKENIZER = Tokenizer(path=path)
    return _TOKENIZER.tokenize(text)


class Tokenizer:
    def __init__(self, path: str, n_pad: int=21) -> None:
        self.path = path
        self.n_pad = n_pad
        self.load_model(self.path)

    def load_model(self, path: str):
        self.path = path
        self.model = ort.InferenceSession(self.path)
    
    def tokenize(self, text: str) -> List[str]:
        self.x_char, self.x_type = create_feature_array(text, n_pad=self.n_pad)
        self.x_char = self.x_char.astype(np.float32)
        self.x_type= self.x_type.astype(np.float32)
        self.outputs = self.model.run(
            None,
            {
                'input_1': self.x_char,
                'input_2': self.x_type
            }
        )
        self.y_predict = (self.outputs[0].ravel() > 0.5).astype(int)
        self.word_end = self.y_predict[1:].tolist() + [1]
        self.tokens = []
        self.word = ''
        for char, w_e in zip(text, self.word_end):
            self.word += char
            if w_e:
                self.tokens.append(self.word)
                self.word = ''
        return self.tokens
