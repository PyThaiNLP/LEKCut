# -*- coding: utf-8 -*-
__version__ = "0.1"
from typing import List
import lekcut.deepcut as _deepcut
import lekcut.attacut as _attacut
import lekcut.sefrcut as _sefrcut

_ATTACUT_MODELS = {"attacut-sc", "attacut-c"}
_SEFRCUT_ENGINES = {"sefr-ws1000", "sefr-tnhc", "sefr-best"}


def word_tokenize(text: str, model: str="deepcut", path: str="default", providers: List[str]=None) -> List[str]:
    if model == "deepcut":
        return _deepcut.tokenize(text, path=path, providers=providers)
    if model in _ATTACUT_MODELS:
        return _attacut.tokenize(text, model=model, path=path, providers=providers)
    if model in _SEFRCUT_ENGINES:
        engine = model[len("sefr-"):]
        return _sefrcut.tokenize(text, engine=engine, deepcut_path=path, providers=providers)
    raise NotImplementedError("Not support {} model.".format(model))