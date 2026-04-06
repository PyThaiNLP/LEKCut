# -*- coding: utf-8 -*-
"""
OSKut: Out-of-domain Stacked Cut for Thai Word Segmentation

Limkonchotiwat, P., Phatthiyaphaibun, W., Sarwar, R., Chuangsuwanich, E., &
Nutanong, S. (2021). Handling Cross- and Out-of-Domain Samples in Thai Word
Segmentation. In Findings of the ACL: IJCNLP 2021.

License: MIT License (For Model and Code that come from OSKut's GitHub)
GitHub: https://github.com/mrpeerat/OSKut
"""
from typing import List, Optional

_LOADED_ENGINE: Optional[str] = None


def tokenize(text: str, engine: str = "ws", k: int = 1) -> List[str]:
    """Tokenize *text* using the OSKut model.

    Parameters
    ----------
    text:
        Input Thai text.
    engine:
        OSKut engine to use. Options: ``"ws"`` (default),
        ``"ws-augment-60p"``, ``"tnhc"``, ``"best"``, ``"scads"``,
        ``"tl-deepcut-ws"``, ``"tl-deepcut-tnhc"``, ``"deepcut"``.
    k:
        Percentage of characters to refine (1–100). The special default
        value of ``1`` is a sentinel that lets OSKut automatically select
        an appropriate percentage based on the engine (e.g. 33 for ``ws``,
        100 for ``tnhc``). Pass any integer from 2 to 100 to override this
        automatic selection.
    """
    global _LOADED_ENGINE

    try:
        import oskut
    except ImportError as e:
        raise ImportError(
            "The 'OSKut' package is required for the oskut model. "
            "Install it with: pip install OSKut"
        ) from e

    if _LOADED_ENGINE != engine:
        oskut.load_model(engine=engine)
        _LOADED_ENGINE = engine

    return oskut.OSKut(text, k=k)
