# -*- coding: utf-8 -*-
__version__ = "1.0.0"
from typing import List, Optional
import lekcut.deepcut as _deepcut
import lekcut.attacut as _attacut
import lekcut.oskut as _oskut
import lekcut.sefrcut as _sefrcut

_ATTACUT_MODELS = {"attacut-sc", "attacut-c"}
_SEFRCUT_ENGINES = {"sefr-ws1000", "sefr-tnhc", "sefr-best"}


def word_tokenize(
    text: str,
    model: str = "deepcut",
    path: str = "default",
    providers: Optional[List[str]] = None,
    engine: str = "ws",
    k: int = 1,
) -> List[str]:
    """Tokenize Thai *text* using the selected model.

    Parameters
    ----------
    text:
        Input Thai text.
    model:
        Model to use. Options: ``"deepcut"`` (default), ``"attacut-sc"``,
        ``"attacut-c"``, ``"oskut"``.
    path:
        Path to a custom ONNX model file (applies to ``deepcut`` and
        ``attacut-*`` models only). Pass ``"default"`` to use the bundled
        model.
    providers:
        ONNX Runtime execution providers (applies to ``deepcut`` and
        ``attacut-*`` models only).
    engine:
        OSKut engine variant (applies to ``"oskut"`` model only). Options:
        ``"ws"`` (default), ``"ws-augment-60p"``, ``"tnhc"``, ``"scads"``,
        ``"tl-deepcut-ws"``, ``"tl-deepcut-tnhc"``, ``"deepcut"``.
    k:
        Percentage of characters to refine for OSKut (applies to
        ``"oskut"`` model only). The special default value of ``1`` is a
        sentinel that lets OSKut automatically select an appropriate
        percentage based on the engine. Pass any integer from 2 to 100 to
        override this automatic selection.
    """
    if model == "deepcut":
        return _deepcut.tokenize(text, path=path, providers=providers)
    if model in _ATTACUT_MODELS:
        return _attacut.tokenize(text, model=model, path=path, providers=providers)
    if model == "oskut":
        return _oskut.tokenize(text, engine=engine, k=k, providers=providers)
    if model in _SEFRCUT_ENGINES:
        engine = model[len("sefr-"):]
        return _sefrcut.tokenize(text, engine=engine, deepcut_path=path, providers=providers)
    raise NotImplementedError("Not support {} model.".format(model))
