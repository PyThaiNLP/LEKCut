# -*- coding: utf-8 -*-
"""
SEFR CUT: Domain Adaptation of Thai Word Segmentation Models using Stacked Ensemble

Limkonchotiwat, P., Phatthiyaphaibun, W., Sarwar, R., Chuangsuwanich, E., &
Nutanong, S. (2020). Domain Adaptation of Thai Word Segmentation Models using
Stacked Ensemble. In Proceedings of EMNLP 2020.

License: MIT License (For Model and Code that come from SEFR CUT's GitHub)
GitHub: https://github.com/mrpeerat/SEFR_CUT
"""
import math
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from lekcut.deepcut import create_feature_array
from lekcut.model import get_path

# ── Character type mapping for CRF features ──────────────────────────────────
# Adapted from SEFR CUT's extract_features.py

_CHARTYPE_TAGS = [
    ("c", "กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ"),
    ("n", "ฅฉผฟฌหฮ"),
    ("v", "ะาำิีืึุูๅ็"),
    ("w", "เแโใไ"),
    ("s", "ๆฯ.์"),
    ("a", "ฯๆ๏"),
    ("t", "่้๊๋"),
    ("d", "0123456789๑๒๓๔๕๖๗๘๙"),
    ("b", "$฿"),
    ("q", "'\""),
    ("o", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    ("z", " \u00a0"),
]


def _get_ctype(c: str) -> str:
    for tag, chars in _CHARTYPE_TAGS:
        if c in chars:
            return tag
    return "x"


# ── Aho-Corasick automaton for dictionary features ────────────────────────────

_AUTOMATON = None


def _get_automaton():
    global _AUTOMATON
    if _AUTOMATON is not None:
        return _AUTOMATON
    try:
        import ahocorasick
    except ImportError as e:
        raise ImportError(
            "The 'pyahocorasick' package is required for SEFR CUT. "
            "Install it with: pip install pyahocorasick"
        ) from e
    dict_path = get_path("sefr_words.txt")
    automaton = ahocorasick.Automaton()
    with open(dict_path, "r", encoding="utf-8-sig") as f:
        for word in f.read().strip().split("\n"):
            automaton.add_word(word, len(word))
    automaton.make_automaton()
    _AUTOMATON = automaton
    return _AUTOMATON


# ── CRF feature extraction ────────────────────────────────────────────────────


def _extract_features_crf(
    doc: str,
    y_entropy: List[float],
    y_prob: List[List[float]],
) -> List[List[dict]]:
    """Extract per-character CRF feature dicts for *doc*."""
    automaton = _get_automaton()

    dict_start_boundaries: set = set()
    dict_end_boundaries: set = set()
    for end_index, length in automaton.iter(doc):
        start_index = end_index - length + 1
        dict_start_boundaries.add(start_index)
        dict_end_boundaries.add(end_index)

    doc_features: List[List[dict]] = []
    n = len(doc)
    back_ward = 4
    for_ward = 2

    for i, char in enumerate(doc):
        feat: dict = {
            "bias": "b",
            "char": char,
            "entropy": y_entropy[i],
            "prob": y_prob[i][1],
            "start": (i == 0),
            "end": (i == n - 1),
        }

        # Backward context
        if i < back_ward:
            for index in range(back_ward - i, 0, -1):
                feat[f"char_[-{index + i}]"] = " "
                feat[f"ctype[-{index + i}]"] = _get_ctype(" ")
        for index in range(back_ward):
            try:
                feat[f"char_[-{index + 1}]"] = doc[i - index - 1]
                feat[f"ctype[-{index + 1}]"] = _get_ctype(doc[i - index - 1])
            except IndexError:
                continue

        # Forward context
        text_fwd = doc[i + 1 : i + for_ward + 1]
        while len(text_fwd) < for_ward:
            text_fwd += " "
        for index, c in enumerate(text_fwd):
            feat[f"char_[+{index + 1}]"] = c
            feat[f"ctype[+{index + 1}]"] = _get_ctype(c)

        feat["dict_start"] = i in dict_start_boundaries
        feat["dict_end"] = i in dict_end_boundaries

        doc_features.append([feat])

    return doc_features


# ── Numerical helpers ─────────────────────────────────────────────────────────


def _normalize_l2(arr: np.ndarray) -> np.ndarray:
    """L2-normalise each row of *arr*."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _row_entropy(arr: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row (probability distribution)."""
    clipped = np.clip(arr, 1e-10, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _top_k_percent_indices(k: int, entropy_list: List[float]) -> List[int]:
    """Return indices of the top-k% highest-entropy characters."""
    ranking_times = int(len(entropy_list) * (k / 100))
    entropy_arr = np.array(entropy_list, dtype=np.float64)
    indices: List[int] = []
    for _ in range(ranking_times):
        idx = int(np.argmax(entropy_arr))
        entropy_arr[idx] = -math.inf
        indices.append(idx)
    return indices


# ── Core SEFR CUT algorithm ───────────────────────────────────────────────────


def _apply_crf(tagger, doc, y_pred, y_entropy, y_prob, entropy_index):
    """Refine *y_pred* at high-entropy positions using the CRF *tagger*."""
    result = list(y_pred)
    features = _extract_features_crf(doc, y_entropy, y_prob)
    for idx in entropy_index:
        crf_tag = tagger.tag(features[idx])
        result[idx] = int(crf_tag[0])
    return result


def _preds_to_words(text: str, y_pred: List[int]) -> List[str]:
    """Reconstruct word tokens from per-character boundary predictions.

    Convention (same as SEFR CUT): ``y_pred[i] == 1`` means character *i*
    is the **first** character of a new word.
    """
    tokens: List[str] = []
    word: str = ""
    for char, pred in zip(text, y_pred):
        if pred == 1 and word:
            tokens.append(word)
            word = char
        else:
            word += char
    if word:
        tokens.append(word)
    return tokens


# ── Provider comparison helper ────────────────────────────────────────────────


def _compare_providers(
    p1: Optional[List[str]], p2: Optional[List[str]]
) -> bool:
    if p1 is None and p2 is None:
        return True
    if p1 is None or p2 is None:
        return False
    return p1 == p2


# ── Tokenizer class ───────────────────────────────────────────────────────────

_TOKENIZER_CACHE: dict = {}

_DEFAULT_K = {"ws1000": 100, "tnhc": 36, "best": 5}
_VALID_ENGINES = frozenset(_DEFAULT_K.keys())


class SefrCutTokenizer:
    """Tokenizer backed by an ONNX DeepCut base model and a CRF refiner.

    Parameters
    ----------
    engine:
        One of ``"ws1000"``, ``"tnhc"``, or ``"best"``.
    deepcut_path:
        Path to the DeepCut ONNX model.  Pass ``"default"`` to use the
        bundled model.
    providers:
        ONNX Runtime execution providers.
    """

    def __init__(
        self,
        engine: str = "ws1000",
        deepcut_path: str = "default",
        providers: Optional[List[str]] = None,
    ) -> None:
        if engine not in _VALID_ENGINES:
            raise ValueError(
                f"Unknown SEFR CUT engine '{engine}'. "
                f"Choose one of: {', '.join(sorted(_VALID_ENGINES))}."
            )
        self.engine = engine
        self.providers = providers
        self.default_k = _DEFAULT_K[engine]

        # ONNX DeepCut base model
        if deepcut_path == "default":
            deepcut_path = get_path("deepcut.onnx")
        self.deepcut_path = deepcut_path
        if providers is None:
            self._deepcut_model = ort.InferenceSession(deepcut_path)
        else:
            self._deepcut_model = ort.InferenceSession(
                deepcut_path, providers=providers
            )

        # pycrfsuite CRF refiner
        try:
            import pycrfsuite
        except ImportError as e:
            raise ImportError(
                "The 'python-crfsuite' package is required for SEFR CUT. "
                "Install it with: pip install python-crfsuite"
            ) from e
        crf_path = get_path(f"sefr_{engine}.model")
        self._crf_tagger = pycrfsuite.Tagger()
        self._crf_tagger.open(crf_path)

    # ------------------------------------------------------------------

    def _base_probs(self, text: str) -> np.ndarray:
        """Run the ONNX DeepCut model and return raw sigmoid probabilities.

        The probability at position *i* represents how likely character *i*
        is the **start** of a new word.
        """
        x_char, x_type = create_feature_array(text, n_pad=21)
        x_char = x_char.astype(np.float32)
        x_type = x_type.astype(np.float32)
        raw = self._deepcut_model.run(
            None, {"input_1": x_char, "input_2": x_type}
        )[0].ravel()
        return raw  # shape: (len(text),)

    def tokenize(self, text: str, k: int = 0) -> List[str]:
        """Tokenize *text*.

        Parameters
        ----------
        text:
            Input Thai text.
        k:
            Percentage of highest-entropy characters to refine with the CRF
            (0 = use the engine default).
        """
        if not text:
            return []
        if k == 0:
            k = self.default_k

        # Base predictions from ONNX DeepCut
        probs = self._base_probs(text)  # (n,) word-start probabilities

        # Build [P(non-start), P(start)] for each char, same as SEFR CUT
        y_prob_arr = np.stack([1.0 - probs, probs], axis=1)  # (n, 2)
        y_prob_norm = _normalize_l2(y_prob_arr)
        y_entropy = _row_entropy(y_prob_norm).tolist()
        y_pred = (probs > 0.5).astype(int).tolist()
        y_prob_list = y_prob_arr.tolist()

        # Identify high-entropy characters and refine with CRF
        entropy_index = _top_k_percent_indices(k, y_entropy)
        y_pred_refined = _apply_crf(
            self._crf_tagger, text, y_pred, y_entropy, y_prob_list,
            entropy_index,
        )

        return _preds_to_words(text, y_pred_refined)


# ── Module-level convenience function ────────────────────────────────────────


def tokenize(
    text: str,
    engine: str = "ws1000",
    deepcut_path: str = "default",
    providers: Optional[List[str]] = None,
    k: int = 0,
) -> List[str]:
    """Tokenize Thai *text* using SEFR CUT.

    Parameters
    ----------
    text:
        Input Thai text.
    engine:
        CRF model variant: ``"ws1000"`` (Wisesight-1000, default),
        ``"tnhc"`` (TNHC), or ``"best"`` (BEST-2010).
    deepcut_path:
        Path to a custom DeepCut ONNX model.  Use ``"default"`` for the
        bundled model.
    providers:
        ONNX Runtime execution providers.
    k:
        Percentage of highest-entropy characters to refine with the CRF
        (0 = use engine default: ws1000→100, tnhc→36, best→5).
    """
    global _TOKENIZER_CACHE

    cache_key = (engine, deepcut_path, tuple(providers) if providers else None)
    cached = _TOKENIZER_CACHE.get(cache_key)
    if cached is None or not _compare_providers(cached.providers, providers):
        _TOKENIZER_CACHE[cache_key] = SefrCutTokenizer(
            engine=engine,
            deepcut_path=deepcut_path,
            providers=providers,
        )
    return _TOKENIZER_CACHE[cache_key].tokenize(text, k=k)
