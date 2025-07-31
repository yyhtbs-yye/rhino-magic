import os
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any

class ScanFiles:

    def __init__(self, depth):
        
        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")
        self.depth = depth

        # Map depth -> scanning strategy
        self._scan_fn: Callable[[Path, str], Iterable[Path]] = (
            self._scan_1_level if depth == 1 else self._scan_2_level
        )

    def __call__(self, folder_paths, extensions, data_prefix) -> Dict[str, Any]:

        img_paths = {}

        for key in extensions.keys():
            img_paths[key] = []
            # ---- resolve folder ------------------------------------------------
            folder_path = folder_paths.get(key)
            if not folder_path:
                raise KeyError(f"Missing folder path for key '{key}'")

            prefix = os.path.join(folder_path, data_prefix.get(key, ""))
            base = Path(prefix)

            # ---- collect paths --------------------------------------------------
            for ext in extensions[key]:
                img_paths[key].extend(
                    str(p) for p in self._scan_fn(base, ext)
                )

            # ---- sanity check ---------------------------------------------------
            if not img_paths[key]:
                raise ValueError(f"No images found in '{folder_path}'")

        return img_paths

    @staticmethod
    def _scan_1_level(base: Path, ext: str) -> List[Path]:
        """Return files <base>/<file>.<ext>  (no sub-dirs)."""
        return list(base.glob(f"*{ext}"))

    @staticmethod
    def _scan_2_level(base: Path, ext: str) -> List[Path]:
        """Return files <base>/<sub>/<file>.<ext>  (exactly two levels)."""
        return list(base.glob(f"*/*{ext}"))

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

class ReshapeByFilename:
    """
    Reshape a dict of lists of paths into a list of dicts grouped by filename.
    """

    def __call__(self, img_paths: Dict[str, List[str]]) -> List[Dict[str, str]]:
        # Step 1: Build a mapping from filename -> {key: path}
        grouped: Dict[str, Dict[str, str]] = defaultdict(dict)

        for key, paths in img_paths.items():
            for path_str in paths:
                path = Path(path_str)
                filename = path.stem
                grouped[filename][f"{key}_path"] = path_str

        # Step 2: Filter out incomplete groups (optional, depending on strictness)
        reshaped = [
            group for group in grouped.values()
            if len(group) == len(img_paths)
        ]

        return reshaped
    
