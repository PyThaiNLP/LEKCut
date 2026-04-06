# -*- coding: utf-8 -*-
__version__ = "0.1"
from typing import List
import lekcut.deepcut as _deepcut
import lekcut.attacut as _attacut

_ATTACUT_MODELS = {"attacut-sc", "attacut-c"}


def word_tokenize(text: str, model: str="deepcut", path: str="default", providers: List[str]=None) -> List[str]:
    if model == "deepcut":
        return _deepcut.tokenize(text, path=path, providers=providers)
    if model in _ATTACUT_MODELS:
        return _attacut.tokenize(text, model=model, path=path, providers=providers)
    raise NotImplementedError("Not support {} model.".format(model))