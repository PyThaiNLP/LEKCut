# -*- coding: utf-8 -*-
"""
OSKut: Out-of-domain Stacked Cut for Thai Word Segmentation

Limkonchotiwat, P., Phatthiyaphaibun, W., Sarwar, R., Chuangsuwanich, E., &
Nutanong, S. (2021). Handling Cross- and Out-of-Domain Samples in Thai Word
Segmentation. In Findings of the ACL: IJCNLP 2021.

License: MIT License (For Model and Code that come from OSKut's GitHub)
GitHub: https://github.com/mrpeerat/OSKut
"""
import operator
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from lekcut.model import get_path

# ---------------------------------------------------------------------------
# Character / type vocabularies (must match OSKut's training preprocessing)
# ---------------------------------------------------------------------------

_CHARS = [
    '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
    ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'other', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z', '}', '~',
    'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
    'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท',
    'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ',
    'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ฯ', 'ะ', 'ั', 'า',
    'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ', 'เ', 'แ', 'โ', 'ใ', 'ไ',
    'ๅ', 'ๆ', '็', '่', '้', '๊', '๋', '์', 'ํ', '๐', '๑', '๒', '๓',
    '๔', '๕', '๖', '๗', '๘', '๙', '\u2018', '\u2019', '\ufeff',
]
_CHARS_MAP = {v: k for k, v in enumerate(_CHARS)}

_CHAR_TYPE = {
    'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    'ฅฉผฟฌหฝฮฤ': 'n',
    'ัะาำิีืึุู': 'v',
    'เแโใไ': 'w',
    '่้๊๋็': 't',
    '์ๆฯ.': 's',
    '0123456789๑๒๓๔๕๖๗๘๙': 'd',
    '"': 'q',
    "'": 'q',
    '\u2018': 'q',
    '\u2019': 'q',
    ' ': 'p',
    '<>`~๐;:-({)},./+*/-?!@#$%^&=][': 'p',
    'abcdefghijklmnopqrstuvwxyz': 's_e',
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e',
}
_CHAR_TYPE_FLATTEN: dict = {}
for _ks, _v in _CHAR_TYPE.items():
    for _k in _ks:
        _CHAR_TYPE_FLATTEN[_k] = _v

_CHAR_TYPES = ['b_e', 'c', 'd', 'n', 'o', 'p', 'q', 's', 's_e', 't', 'v', 'w']
_CHAR_TYPES_MAP = {v: k for k, v in enumerate(_CHAR_TYPES)}

# Engine configuration: maps engine name → (lstm_nodes, attention_nodes)
# Engines whose value is None use the single tl-deepcut ONNX model directly.
_ENGINE_CONFIGS = {
    'ws':             (192, 96),
    'tnhc':           (192, 160),
    'scads':          (224, 96),
    'ws-augment-60p': (192, 32),
    'tl-deepcut-ws':  None,
    'tl-deepcut-tnhc': None,
    'deepcut':        None,
}

# Default k values (% of chars to refine) matching OSKut's heuristics.
_DEFAULT_K = {
    'ws':             33,
    'tnhc':           100,
    'scads':          100,
    'ws-augment-60p': 100,
}


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _create_feature_array(text: str, n_pad: int = 21):
    """Create per-character context-window feature arrays (char indices + type indices)."""
    n = len(text)
    n_pad_2 = (n_pad - 1) // 2
    text_pad = [' '] * n_pad_2 + list(text) + [' '] * n_pad_2
    x_char, x_type = [], []
    for i in range(n_pad_2, n_pad_2 + n):
        window = (
            text_pad[i + 1: i + n_pad_2 + 1]
            + list(reversed(text_pad[i - n_pad_2: i]))
            + [text_pad[i]]
        )
        x_char.append([_CHARS_MAP.get(c, 80) for c in window])
        x_type.append([
            _CHAR_TYPES_MAP.get(_CHAR_TYPE_FLATTEN.get(c, 'o'), 4)
            for c in window
        ])
    return (
        np.array(x_char, dtype=np.float32),
        np.array(x_type, dtype=np.float32),
    )


def _row_l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalise each row (matching sklearn.preprocessing.Normalizer)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row (matching scipy.stats.entropy with base=None)."""
    p = np.clip(p, 1e-15, None)
    return -np.sum(p * np.log(p), axis=1)


def _build_automaton(words_path: str):
    """Build an Aho-Corasick automaton from the OSKut word list."""
    try:
        import ahocorasick
    except ImportError as exc:
        raise ImportError(
            "The 'pyahocorasick' package is required for the oskut model. "
            "Install it with: pip install pyahocorasick"
        ) from exc
    automaton = ahocorasick.Automaton()
    with open(words_path, encoding='utf-8-sig') as fh:
        for word in fh.read().strip().split('\n'):
            automaton.add_word(word, len(word))
    automaton.make_automaton()
    return automaton


def _dict_boundaries(text: str, automaton) -> tuple:
    """Return (start_set, end_set) of character positions that are dict-word boundaries."""
    starts: set = set()
    ends: set = set()
    for end_idx, length in automaton.iter(text):
        starts.add(end_idx - length + 1)
        ends.add(end_idx)
    return starts, ends


def _make_additional_features(
    text: str,
    entropy_vals: np.ndarray,
    prob_vals: np.ndarray,
    automaton,
    maxlen: int = 21,
) -> np.ndarray:
    """Build the per-character additional feature matrix for the LSTM model."""
    starts, ends = _dict_boundaries(text, automaton)
    rows = []
    for i, _ in enumerate(text):
        row = [
            float(entropy_vals[i]),
            float(prob_vals[i]),
            1.0 if i in starts else 0.0,
            1.0 if i in ends else 0.0,
        ]
        # Pad to maxlen with zeros (post-padding, matching Keras pad_sequences)
        row += [0.0] * (maxlen - len(row))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------

def _compare_providers(p1: Optional[List[str]], p2: Optional[List[str]]) -> bool:
    if p1 is None and p2 is None:
        return True
    if p1 is None or p2 is None:
        return False
    return p1 == p2


class OskutTokenizer:
    """ONNX-based OSKut tokenizer. Requires no TensorFlow dependency."""

    def __init__(
        self,
        engine: str = "ws",
        providers: Optional[List[str]] = None,
    ) -> None:
        if engine not in _ENGINE_CONFIGS:
            raise ValueError(
                f"Unknown OSKut engine '{engine}'. "
                f"Choose one of: {sorted(_ENGINE_CONFIGS)}"
            )
        self.engine = engine
        self.providers = providers
        self._is_tl = engine.startswith('tl-deepcut') or engine == 'deepcut'

        def _session(path: str) -> ort.InferenceSession:
            if providers is None:
                return ort.InferenceSession(path)
            return ort.InferenceSession(path, providers=providers)

        if self._is_tl:
            # Single CNN model (tl-deepcut or plain deepcut)
            if engine == 'deepcut':
                model_file = 'oskut-deepcut.onnx'
            else:
                model_file = f'oskut-{engine}.onnx'
            self._model = _session(get_path(model_file))
        else:
            # Two-stage: baseline deepcut + LSTM+Attention refinement
            self._baseline = _session(get_path('oskut-deepcut.onnx'))
            self._refiner = _session(get_path(f'oskut-{engine}.onnx'))
            self._automaton = _build_automaton(get_path('oskut-words.txt'))

    def tokenize(self, text: str, k: int = 1) -> List[str]:
        if not text:
            return []

        x_char, x_type = _create_feature_array(text)

        if self._is_tl:
            return self._tokenize_tl(text, x_char, x_type)

        # Resolve the effective k value
        if k == 1:
            effective_k = _DEFAULT_K.get(self.engine, 33)
        else:
            effective_k = k

        return self._tokenize_stacked(text, x_char, x_type, effective_k)

    # -- tl-deepcut / plain-deepcut path ------------------------------------

    def _tokenize_tl(
        self, text: str, x_char: np.ndarray, x_type: np.ndarray
    ) -> List[str]:
        inputs = self._model.get_inputs()
        raw = self._model.run(None, {
            inputs[0].name: x_char,
            inputs[1].name: x_type,
        })[0].ravel()
        word_end = (raw[1:] > 0.5).tolist() + [True]
        return _reconstruct(text, word_end)

    # -- two-stage OSKut path -----------------------------------------------

    def _tokenize_stacked(
        self,
        text: str,
        x_char: np.ndarray,
        x_type: np.ndarray,
        k: int,
    ) -> List[str]:
        # 1. Baseline (deepcut) prediction
        bl_inputs = self._baseline.get_inputs()
        raw_prob = self._baseline.run(None, {
            bl_inputs[0].name: x_char,
            bl_inputs[1].name: x_type,
        })[0].ravel()  # shape (n_chars,), probability of word-start

        # 2. Compute entropy for each character
        prob_2d = np.stack([1.0 - raw_prob, raw_prob], axis=1)
        norm_prob = _row_l2_normalize(prob_2d)
        entropy_vals = _entropy(norm_prob)

        # 3. Initial binary predictions (argmax of [1-p, p])
        y_pred = (raw_prob > 0.5).astype(int).tolist()

        # 4. Select top-k% of characters by entropy
        n_select = int(len(entropy_vals) * (k / 100))
        if n_select == 0:
            return _reconstruct_from_boundary(text, y_pred)

        # Get indices sorted by entropy descending, take top-n_select
        indexed = sorted(enumerate(entropy_vals), key=operator.itemgetter(1), reverse=True)
        refine_idx = sorted([idx for idx, _ in indexed[:n_select]])

        # 5. Build additional features for ALL characters (then select rows)
        additional = _make_additional_features(
            text, entropy_vals, raw_prob, self._automaton
        )

        char_sel = x_char[refine_idx]
        type_sel = x_type[refine_idx]
        add_sel = additional[refine_idx]

        # 6. Run LSTM+Attention refiner on selected chars
        ref_inputs = self._refiner.get_inputs()
        refined = self._refiner.run(None, {
            ref_inputs[0].name: char_sel,
            ref_inputs[1].name: type_sel,
            ref_inputs[2].name: add_sel,
        })[0].ravel()
        refined_pred = (refined > 0.5).astype(int).tolist()

        # 7. Merge refined predictions back
        for pos, pred in zip(refine_idx, refined_pred):
            y_pred[pos] = pred

        return _reconstruct_from_boundary(text, y_pred)


# ---------------------------------------------------------------------------
# Word reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(text: str, word_end: list) -> List[str]:
    """Rebuild word list from per-character 'is word end' flags."""
    tokens, word = [], ''
    for ch, end in zip(text, word_end):
        word += ch
        if end:
            tokens.append(word)
            word = ''
    if word:
        tokens.append(word)
    return tokens


def _reconstruct_from_boundary(text: str, y_pred: list) -> List[str]:
    """Rebuild word list from OSKut-style boundary predictions.

    OSKut marks the *first* character of a new word with ``1``, so:
    ``1`` at position ``i`` means character ``i`` starts a new token.
    """
    tokens, word = [], ''
    for i, (ch, pred) in enumerate(zip(text, y_pred)):
        if i > 0 and pred == 1:
            tokens.append(word)
            word = ''
        word += ch
    if word:
        tokens.append(word)
    return tokens


# ---------------------------------------------------------------------------
# Module-level cache + public entry point
# ---------------------------------------------------------------------------

_TOKENIZER: Optional[OskutTokenizer] = None


def tokenize(text: str, engine: str = "ws", k: int = 1,
             providers: Optional[List[str]] = None) -> List[str]:
    """Tokenize *text* using the ONNX-based OSKut model.

    Parameters
    ----------
    text:
        Input Thai text.
    engine:
        OSKut engine variant. Options: ``"ws"`` (default),
        ``"ws-augment-60p"``, ``"tnhc"``, ``"scads"``,
        ``"tl-deepcut-ws"``, ``"tl-deepcut-tnhc"``, ``"deepcut"``.
    k:
        Percentage of characters to refine (1–100). The special default
        value of ``1`` is a sentinel that lets OSKut automatically select
        an appropriate percentage based on the engine (e.g. 33 for ``ws``,
        100 for ``tnhc``). Pass any integer from 2 to 100 to override this
        automatic selection.
    providers:
        ONNX Runtime execution providers (e.g.
        ``["CUDAExecutionProvider", "CPUExecutionProvider"]``).
    """
    global _TOKENIZER
    if (
        _TOKENIZER is None
        or _TOKENIZER.engine != engine
        or not _compare_providers(_TOKENIZER.providers, providers)
    ):
        _TOKENIZER = OskutTokenizer(engine=engine, providers=providers)
    return _TOKENIZER.tokenize(text, k=k)

