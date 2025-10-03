import os
from pathlib import Path
from torch.utils.data import Dataset
from trainer.torch_datasets import transforms

class UnpairedHalvesImageDataset(Dataset):
    """
    Unpaired image dataset that accepts exactly TWO keys (folders).
    It builds an index so that for every __getitem__(i), the sample combines:
      - key1's i-th image (by sorted common basename)
      - key2's image from the *other* half (later half first, then first half),
    ensuring you never get the same-basename pair at the same index.

    Notes
    -----
    - Uses the intersection of basenames across the two folders (like BasicImageDataset).
    - If the number of common images is 1, we raise (cannot unpair).
    - For odd counts, we still "rotate by floor(N/2)" which maintains unpaired indices.
    """

    def __init__(self, **dataset_config):
        super().__init__()

        # Expect the same config keys as BasicImageDataset
        self.folder_paths = dataset_config.get('folder_paths', {})
        self.data_prefix  = dataset_config.get('data_prefix', {})
        pipeline_cfg      = dataset_config.get('pipeline', [])

        # Build transform pipeline (same style)
        self.transform_pipeline = self._build_pipeline(pipeline_cfg)

        # Scan & build unpaired index
        self.image_paths = self._scan_images()  # list of dicts, each dict has "<key>_path" entries

    def _scan_images(self):
        """Scan exactly two keys, find common basenames, and build a half-swapped unpaired index."""
        if not self.folder_paths or len(self.folder_paths) != 2:
            raise ValueError("This dataset requires exactly TWO keys in 'folder_paths' (e.g., {'A': '...', 'B': '...'}).")

        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
        paths_by_key = {}
        base_names_by_key = {}

        # Preserve insertion order of keys (Python 3.7+ dicts are ordered)
        keys = list(self.folder_paths.keys())
        k1, k2 = keys[0], keys[1]

        # Gather images & basenames for each key
        for key in keys:
            folder_path = self.folder_paths[key]
            path_prefix = os.path.join(folder_path, self.data_prefix.get(key, ''))
            image_paths = []
            for ext in extensions:
                image_paths.extend([str(p) for p in Path(path_prefix).glob(f'**/*{ext}')])

            if len(image_paths) == 0:
                raise ValueError(f"No images found in: {folder_path}")

            paths_by_key[key] = image_paths
            base_names_by_key[key] = {os.path.basename(p): p for p in image_paths}
            print(f"[{key}] Found {len(image_paths)} images in {folder_path}")

        # Intersection of basenames across the two folders
        common = sorted(set(base_names_by_key[k1].keys()) & set(base_names_by_key[k2].keys()))
        if not common:
            raise ValueError("No common images (by basename) found across the two folders.")
        if len(common) < 2:
            raise ValueError("Need at least 2 common images to ensure unpaired sampling.")

        print(f"Found {len(common)} common images across both folders.")

        # Build *aligned* lists by sorted common basenames
        k1_list = [base_names_by_key[k1][bn] for bn in common]
        k2_list = [base_names_by_key[k2][bn] for bn in common]

        # Swap halves for k2 so indices never align on the same basename.
        # For N images, rotate by N//2: k2[half:] + k2[:half].
        N = len(common)
        half = N // 2
        k2_unpaired = k2_list[half:] + k2_list[:half]

        # Create list of records with "<key>_path" keys (same naming style as BasicImageDataset)
        # index i uses k1_list[i] with k2_unpaired[i]
        samples = []
        for i in range(N):
            samples.append({
                f"{k1}_path": k1_list[i],
                f"{k2}_path": k2_unpaired[i],
            })

        # Optional sanity: verify no basename collision at the same index
        # (not strictly necessary, but helpful during development)
        # for i in range(N):
        #     assert os.path.basename(k1_list[i]) != os.path.basename(k2_unpaired[i]), \
        #            "Unpaired mapping failed; got a paired basename at the same index."

        return samples

    def _build_pipeline(self, pipeline_cfg):
        """Build the data processing pipeline using getattr for dynamic class loading."""
        transforms_list = []
        for transform_cfg in pipeline_cfg:
            cfg = transform_cfg.copy()
            ttype = cfg.pop('type')
            tclass = getattr(transforms, ttype)
            transforms_list.append(tclass(**cfg))
        return transforms_list

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data = self.image_paths[idx]  # already a dict with "<key>_path" fields
        for transform in self.transform_pipeline:
            data = transform(data)
        return data
