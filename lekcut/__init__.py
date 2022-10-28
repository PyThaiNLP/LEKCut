# -*- coding: utf-8 -*-
__version__ = "0.1"
from typing import List
from lekcut.deepcut import tokenize


def word_tokenize(text: str, model: str="deepcut", path: str="default") -> List[str]:
    if model != "deepcut":
        raise NotImplementedError("Not support {} model.".format(model))
    return tokenize(text, path=path)