# -*- coding: utf-8 -*-
import os
from importlib import resources
import lekcut

def get_path(file: str) -> str:
    # 1. Prevent Directory Traversal
    # Ensures the filename doesn't contain '..' or absolute paths
    basename = os.path.basename(file)
    
    # 2. Use importlib to locate the resource within the 'lekcut.model' sub-package
    # 'files()' is available in Python 3.9+
    with resources.as_file(resources.files(lekcut) / "model" / basename) as path:
        return str(path)
