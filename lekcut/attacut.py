# -*- coding: utf-8 -*-
"""
AttaCut: Fast and Accurate Neural Thai Word Segmenter

Wiriyathammabhum, P., Nararatwong, R., Netisopakul, P., & Ratanapitak, P.
(2019). AttaCut: A Fast Thai Word Tokenizer Combining Sound Boundary and
Character Clusters. arXiv:1911.07948.

License: MIT License (For Model and Code that come from AttaCut's GitHub)
GitHub: https://github.com/PyThaiNLP/attacut
"""
import json
import re
import string
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from lekcut.model import get_path

ARABIC_RX = re.compile(r"[A-Za-z]+")
NUMBER_RX = re.compile(r"[0-9,]+")


def _character2ix(ch2ix: dict, character: str) -> int:
    if character == "":
        return ch2ix["<PAD>"]
    elif character in string.punctuation:
        return ch2ix.get("<PUNC>", ch2ix["<UNK>"])
    return ch2ix.get(character, ch2ix["<UNK>"])


def _syllable2token(syllable: str) -> str:
    if ARABIC_RX.match(syllable):
        return "<ENGLISH>"
    elif NUMBER_RX.match(syllable):
        return "<NUMBER>"
    return syllable


def _syllable2ix(sy2ix: dict, syllable: str) -> int:
    token = _syllable2token(syllable)
    return sy2ix.get(token, sy2ix["<UNK>"])


def _syllable_tokenize(txt: str) -> List[str]:
    """Syllable tokenization using the ssg library."""
    try:
        import ssg
    except ImportError as e:
        raise ImportError(
            "The 'ssg' package is required for the attacut-sc model. "
            "Install it with: pip install ssg"
        ) from e

    seps = txt.split(" ")
    syllables = []
    for i, s in enumerate(seps):
        syllables.extend(ssg.syllable_tokenize(s))
        if i < len(seps) - 1:
            syllables.append(" ")
    return syllables


def _find_words_from_preds(tokens: List[str], preds) -> List[str]:
    """Reconstruct word tokens from per-character boundary predictions."""
    curr_word = tokens[0]
    words = []
    for char, pred in zip(tokens[1:], preds[1:]):
        if pred == 0:
            curr_word += char
        else:
            words.append(curr_word)
            curr_word = char
    words.append(curr_word)
    return words


def _sigmoid(x):
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x64, -500.0, 500.0)))


def _compare_providers(p1: Optional[List[str]], p2: Optional[List[str]]) -> bool:
    if p1 is None and p2 is None:
        return True
    if p1 is None or p2 is None:
        return False
    return p1 == p2


# ── Per-model module-level tokenizer caches ──────────────────────────────────
_TOKENIZER_SC = None
_TOKENIZER_C = None


def tokenize(
    text: str,
    model: str = "attacut-sc",
    path: str = "default",
    providers: Optional[List[str]] = None,
) -> List[str]:
    """Tokenize *text* using the requested AttaCut ONNX model.

    Parameters
    ----------
    text:
        Input Thai text.
    model:
        ``"attacut-sc"`` (syllable + character, default) or
        ``"attacut-c"`` (character only).
    path:
        Path to a custom ONNX model file. Pass ``"default"`` to use the
        bundled model.
    providers:
        ONNX Runtime execution providers (e.g.
        ``["CUDAExecutionProvider", "CPUExecutionProvider"]``).
        Defaults to the ONNX Runtime default (CPU).
    """
    global _TOKENIZER_SC, _TOKENIZER_C

    if model == "attacut-sc":
        if path == "default":
            path = get_path("attacut-sc.onnx")
        if (
            _TOKENIZER_SC is None
            or _TOKENIZER_SC.path != path
            or not _compare_providers(_TOKENIZER_SC.providers, providers)
        ):
            _TOKENIZER_SC = AttacutSCTokenizer(path=path, providers=providers)
        return _TOKENIZER_SC.tokenize(text)

    if model == "attacut-c":
        if path == "default":
            path = get_path("attacut-c.onnx")
        if (
            _TOKENIZER_C is None
            or _TOKENIZER_C.path != path
            or not _compare_providers(_TOKENIZER_C.providers, providers)
        ):
            _TOKENIZER_C = AttacutCTokenizer(path=path, providers=providers)
        return _TOKENIZER_C.tokenize(text)

    raise ValueError(
        f"Unknown AttaCut model variant '{model}'. "
        "Choose 'attacut-sc' or 'attacut-c'."
    )


# ── Base tokenizer ────────────────────────────────────────────────────────────

class _AttacutTokenizer:
    def __init__(
        self,
        path: str,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.path = path
        self.providers = providers
        if providers is None:
            self.model = ort.InferenceSession(path)
        else:
            self.model = ort.InferenceSession(path, providers=providers)

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError


# ── attacut-sc (syllable + character) ────────────────────────────────────────

class AttacutSCTokenizer(_AttacutTokenizer):
    """Tokenizer backed by the ``attacut-sc`` (syllable + character) ONNX model."""

    def __init__(
        self,
        path: str = "default",
        providers: Optional[List[str]] = None,
    ) -> None:
        if path == "default":
            path = get_path("attacut-sc.onnx")
        super().__init__(path, providers)
        with open(get_path("attacut-sc-characters.json"), encoding="utf-8") as f:
            self.ch2ix: dict = json.load(f)
        with open(get_path("attacut-sc-syllables.json"), encoding="utf-8") as f:
            self.sy2ix: dict = json.load(f)

    def _make_feature(self, txt: str):
        syllables = _syllable_tokenize(txt)
        characters = list(txt)
        ch_ix: List[int] = []
        sy_ix: List[int] = []
        for syllable in syllables:
            six = _syllable2ix(self.sy2ix, syllable)
            for ch in syllable:
                ch_ix.append(_character2ix(self.ch2ix, ch))
                sy_ix.append(six)
        features = (
            np.stack((ch_ix, sy_ix), axis=0)
            .reshape(1, 2, -1)
            .astype(np.int64)
        )
        return characters, features

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens, features = self._make_feature(text)
        logits = self.model.run(None, {"input": features})[0]
        preds = (_sigmoid(logits) > 0.5).astype(int)
        return _find_words_from_preds(tokens, preds)


# ── attacut-c (character only) ────────────────────────────────────────────────

class AttacutCTokenizer(_AttacutTokenizer):
    """Tokenizer backed by the ``attacut-c`` (character-only) ONNX model."""

    def __init__(
        self,
        path: str = "default",
        providers: Optional[List[str]] = None,
    ) -> None:
        if path == "default":
            path = get_path("attacut-c.onnx")
        super().__init__(path, providers)
        with open(get_path("attacut-c-characters.json"), encoding="utf-8") as f:
            self.ch2ix: dict = json.load(f)

    def _make_feature(self, txt: str):
        characters = list(txt)
        ch_ix = [_character2ix(self.ch2ix, c) for c in characters]
        features = np.array(ch_ix, dtype=np.int64).reshape(1, -1)
        return characters, features

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens, features = self._make_feature(text)
        logits = self.model.run(None, {"input": features})[0]
        preds = (_sigmoid(logits) > 0.5).astype(int)
        return _find_words_from_preds(tokens, preds)
