# -*- coding: utf-8 -*-
__version__ = "0.1"
from lekcut.deepcut import tokenize


def word_tokenize(text: str, model: str="deepcut", path: str="default"):
    if model != "deepcut":
        raise NotImplementedError("Not support {} model.".format(model))
    return tokenize(text, path=path)