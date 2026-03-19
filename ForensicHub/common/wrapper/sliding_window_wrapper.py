from typing import List, Tuple, Union, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from ForensicHub.registry import register_dataset, build_from_registry, DATASETS


@register_dataset("SlidingWindowWrapper")
class SlidingWindowWrapper(Dataset):
    """Wrapper de dataset qui découpe les images plus grandes qu'une certaine taille
    en patches via une fenêtre glissante avec chevauchement optionnel.

    Reprend la logique de découpe du SplitPreprocessor de data_registry:
    - Calcul du stride à partir du ratio de chevauchement
    - Couverture garantie des bords (ajout de la dernière position)
    - Convention de nommage {basename}_{y}_{x} pour reconstruction

    Usage YAML:
        test_dataset:
          - name: SlidingWindowWrapper
            dataset_name: my_test_sw
            init_config:
              patch_width: 512
              patch_height: 512
              overlapping: 0.5
              dataset:
                name: DocDataset
                init_config:
                  path: /path/to/data
                  get_dct_qtb: True

    Usage programmatique:
        base_ds = DocDataset(path="/path/to/data")
        sw_ds = SlidingWindowWrapper(dataset=base_ds, patch_width=512, patch_height=512)
    """

    def __init__(
        self,
        dataset: Union[Dataset, dict],
        patch_width: int = 512,
        patch_height: int = 512,
        overlapping: float = 0.5,
        common_transform=None,
        post_transform=None,
        post_funcs=None,
        **kwargs,
    ):
        """Initialise le wrapper de fenêtre glissante.

        Args:
            dataset: Dataset source (objet Dataset) ou config dict pour le construire
                     via le registry (format: {"name": "...", "init_config": {...}}).
            patch_width: Largeur des patches en pixels.
            patch_height: Hauteur des patches en pixels.
            overlapping: Ratio de chevauchement entre patches (0.5 = 50%).
            common_transform: Transform à passer au dataset interne (si construit via config).
            post_transform: Post-transform à passer au dataset interne (si construit via config).
            post_funcs: Post-fonctions à passer au dataset interne (si construit via config).
        """
        super().__init__()

        # Construire le dataset interne si c'est un dict de config
        if isinstance(dataset, dict):
            if common_transform is not None:
                dataset.setdefault("init_config", {})["common_transform"] = common_transform
            if post_transform is not None:
                dataset.setdefault("init_config", {})["post_transform"] = post_transform
            if post_funcs is not None:
                dataset.setdefault("init_config", {})["post_funcs"] = post_funcs
            self.dataset = build_from_registry(DATASETS, dataset)
        else:
            self.dataset = dataset

        self.patch_width = int(patch_width)
        self.patch_height = int(patch_height)
        self.overlapping = overlapping

        # Pré-indexation : pour chaque item du dataset source,
        # on calcule les positions des patches.
        self._index_map: List[Tuple[int, int, int]] = []  # (dataset_idx, y, x)
        self._meta: Dict[int, Dict[str, Any]] = {}  # dataset_idx -> meta_info

        self._build_index()

    def _get_image_size(self, idx: int) -> Tuple[int, int]:
        """Récupère la taille (H, W) de l'image à l'index donné.

        Essaie d'abord de lire la taille depuis le fichier (header only)
        pour éviter de déclencher tout le pipeline de transforms.
        """
        # Fast path: lire la taille depuis le chemin du fichier
        ds = self.dataset
        if hasattr(ds, "images") and idx < len(ds.images):
            entry = ds.images[idx]
            img_path = entry[0] if isinstance(entry, (tuple, list)) else entry
            if isinstance(img_path, str):
                try:
                    with PILImage.open(img_path) as im:
                        w, h = im.size
                    return h, w
                except Exception:
                    pass

        # Fallback: charge l'item complet
        item = self.dataset[idx]
        image = item["image"]

        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                _, h, w = image.shape
            else:
                h, w = image.shape
        else:
            if image.ndim == 3:
                h, w, _ = image.shape
            else:
                h, w = image.shape

        return h, w

    def _build_index(self):
        """Pré-calcule l'index des patches pour tout le dataset."""
        print(f"[SlidingWindowWrapper] Pré-indexation de {len(self.dataset)} images...")
        for idx in range(len(self.dataset)):
            h, w = self._get_image_size(idx)

            # Si l'image est plus petite ou égale au patch, on la garde telle quelle
            if w <= self.patch_width and h <= self.patch_height:
                self._index_map.append((idx, -1, -1))  # sentinel: pas de découpe
                self._meta[idx] = {
                    "h": h,
                    "w": w,
                    "split": False,
                }
            else:
                xs, ys = self._compute_patch_positions(w, h)
                self._meta[idx] = {
                    "h": h,
                    "w": w,
                    "split": True,
                    "positions": [(y, x) for y in ys for x in xs],
                }
                for y in ys:
                    for x in xs:
                        self._index_map.append((idx, y, x))

        n_split = sum(1 for m in self._meta.values() if m["split"])
        print(
            f"[SlidingWindowWrapper] {len(self._index_map)} patches totaux "
            f"({n_split}/{len(self.dataset)} images découpées)"
        )

    def _compute_patch_positions(
        self, width: int, height: int
    ) -> Tuple[List[int], List[int]]:
        """Calcule les positions des patches pour couvrir toute l'image.

        Reprend exactement la logique de data_registry/SplitPreprocessor:
        - stride = patch_size // (1 / overlapping)
        - positions générées par range + ajout du bord si nécessaire

        Args:
            width: Largeur de l'image.
            height: Hauteur de l'image.

        Returns:
            Tuple (positions_x, positions_y) des coins supérieurs gauches.
        """
        patch_width = min(self.patch_width, width)
        patch_height = min(self.patch_height, height)

        # Calcul du stride : stride = patch * (1 - overlap)
        stride_x = max(int(patch_width * (1 - self.overlapping)), 1)
        stride_y = max(int(patch_height * (1 - self.overlapping)), 1)

        # Aligner les strides sur des multiples de 8 pour la cohérence DCT
        stride_x = max((stride_x // 8) * 8, 8)
        stride_y = max((stride_y // 8) * 8, 8)

        xs = list(range(0, max(width - patch_width, 0) + 1, stride_x))
        ys = list(range(0, max(height - patch_height, 0) + 1, stride_y))

        # Garantir la couverture des bords (aligné sur 8)
        last_x = ((width - patch_width) // 8) * 8
        last_y = ((height - patch_height) // 8) * 8
        if xs[-1] != last_x:
            xs.append(last_x)
        if ys[-1] != last_y:
            ys.append(last_y)

        return xs, ys

    def _extract_patch(
        self, data: Union[torch.Tensor, np.ndarray], y: int, x: int, is_chw: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extrait un patch d'un tensor ou ndarray.

        Args:
            data: Données source (tensor CHW ou ndarray HW/HWC).
            y: Position Y du coin supérieur gauche.
            x: Position X du coin supérieur gauche.
            is_chw: Si True, le tensor est en format CHW.

        Returns:
            Patch extrait, contiguous en mémoire.
        """
        if isinstance(data, torch.Tensor):
            if is_chw and data.dim() == 3:
                patch = data[:, y : y + self.patch_height, x : x + self.patch_width]
            elif data.dim() == 2:
                patch = data[y : y + self.patch_height, x : x + self.patch_width]
            else:
                patch = data[..., y : y + self.patch_height, x : x + self.patch_width]
            return patch.contiguous()
        else:
            if data.ndim == 3:
                patch = data[y : y + self.patch_height, x : x + self.patch_width, :]
            elif data.ndim == 2:
                patch = data[y : y + self.patch_height, x : x + self.patch_width]
            else:
                patch = data[..., y : y + self.patch_height, x : x + self.patch_width]
            return np.ascontiguousarray(patch)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        dataset_idx, y, x = self._index_map[index]
        item = self.dataset[dataset_idx]
        meta = self._meta[dataset_idx]

        # Pas de découpe nécessaire
        if not meta["split"]:
            item["sw_meta"] = {
                "split": False,
                "origin_h": meta["h"],
                "origin_w": meta["w"],
                "patch_y": 0,
                "patch_x": 0,
                "patch_h": meta["h"],
                "patch_w": meta["w"],
                "original_name": item.get("name", f"img_{dataset_idx}"),
            }
            if "name" in item:
                item["sw_name"] = item["name"]
            return item

        # Découpe en patch
        result = {}
        for key, value in item.items():
            if key in ("image",):
                # Tensor CHW
                result[key] = self._extract_patch(value, y, x, is_chw=True)
            elif key in ("mask", "edge_mask", "shape_mask"):
                # Tensor 1HW ou HW
                if isinstance(value, torch.Tensor) and value.dim() == 3:
                    result[key] = self._extract_patch(value, y, x, is_chw=True)
                else:
                    result[key] = self._extract_patch(value, y, x, is_chw=False)
            elif key in ("dct",):
                # numpy HW — découper spatialement
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    result[key] = self._extract_patch(value, y, x, is_chw=False)
                else:
                    result[key] = value
            elif key in ("qt",):
                # Table de quantification 8x8 — ne pas découper, copier tel quel
                result[key] = value
            elif key in ("origin_shape", "shape"):
                result[key] = torch.tensor([self.patch_height, self.patch_width])
            else:
                # Scalaires, labels, noms, etc. : copier tel quel
                result[key] = value

        # Recalculer le label depuis le mask du patch
        if "mask" in result:
            mask_patch = result["mask"]
            if isinstance(mask_patch, torch.Tensor):
                result["label"] = (mask_patch.sum() != 0).long()
            elif isinstance(mask_patch, np.ndarray):
                result["label"] = torch.tensor(int(mask_patch.sum() != 0)).long()

        # Métadonnées pour la reconstruction
        original_name = item.get("name", f"img_{dataset_idx}")
        result["name"] = f"{original_name}_{y}_{x}"
        result["sw_meta"] = {
            "split": True,
            "origin_h": meta["h"],
            "origin_w": meta["w"],
            "patch_y": y,
            "patch_x": x,
            "patch_h": self.patch_height,
            "patch_w": self.patch_width,
            "original_name": original_name,
        }
        result["sw_name"] = original_name

        return result

    def __str__(self) -> str:
        return (
            f"SlidingWindowWrapper("
            f"patches={len(self)}, "
            f"source_images={len(self.dataset)}, "
            f"patch_size=({self.patch_height}, {self.patch_width}), "
            f"overlapping={self.overlapping})"
        )
