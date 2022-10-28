# -*- coding: utf-8 -*-
import lekcut
import os

lekcut_path = os.path.join(os.path.dirname(lekcut.__file__), "model")

def get_path(file: str) -> str:
    return os.path.join(lekcut_path, file)