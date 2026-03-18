import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def parse_sliding_window_name(name: str) -> Optional[Tuple[str, int, int]]:
    """Parse un nom de patch au format '{basename}_{y}_{x}'.

    Reprend la convention de nommage du SplitPreprocessor de data_registry.

    Args:
        name: Nom du patch (avec ou sans extension).

    Returns:
        Tuple (nom_base, y, x) ou None si le format est invalide.
    """
    # Retirer l'extension si présente
    base = os.path.splitext(os.path.basename(name))[0] if "." in name else name

    parts = base.rsplit("_", 2)
    if len(parts) < 3:
        return None
    try:
        y = int(parts[-2])
        x = int(parts[-1])
        if y > 20000 or x > 20000:
            return None
        original_name = parts[0]
        return original_name, y, x
    except ValueError:
        return None


def merge_predictions(
    predictions: Dict[str, np.ndarray],
    origin_sizes: Dict[str, Tuple[int, int]],
    mode: str = "mean",
) -> Dict[str, np.ndarray]:
    """Fusionne les prédictions de patches en images complètes.

    Reprend la logique de MergePostprocessor de data_registry avec
    les modes de fusion (mean, max, min, overwrite).

    Args:
        predictions: Dictionnaire {nom_patch: prediction_array}.
            Les noms suivent le format '{basename}_{y}_{x}'.
        origin_sizes: Dictionnaire {nom_base: (H, W)} des tailles originales.
        mode: Mode de fusion pour les zones de chevauchement.
            - "mean": Moyenne des valeurs dans les zones de chevauchement.
            - "max": Maximum des valeurs.
            - "min": Minimum des valeurs.
            - "overwrite": Le dernier patch écrase les précédents.

    Returns:
        Dictionnaire {nom_base: image_reconstruite}.
    """
    # Grouper les patches par image de base
    grouped: Dict[str, List[Tuple[int, int, np.ndarray]]] = defaultdict(list)
    ungrouped: Dict[str, np.ndarray] = {}

    for name, pred in predictions.items():
        parsed = parse_sliding_window_name(name)
        if parsed is not None:
            base_name, y, x = parsed
            grouped[base_name].append((y, x, pred))
        else:
            ungrouped[name] = pred

    merged = {}

    # Fusionner chaque groupe
    for base_name, patches in grouped.items():
        if base_name in origin_sizes:
            h, w = origin_sizes[base_name]
        else:
            # Inférer la taille depuis les patches
            h = max(y + p.shape[-2] for y, _, p in patches)
            w = max(x + p.shape[-1] for _, x, p in patches)

        # Déterminer la shape du résultat
        sample_patch = patches[0][2]
        if sample_patch.ndim == 2:
            full = np.zeros((h, w), dtype=np.float64)
            count = np.zeros((h, w), dtype=np.float64)
        elif sample_patch.ndim == 3:
            c = sample_patch.shape[0]
            full = np.zeros((c, h, w), dtype=np.float64)
            count = np.zeros((c, h, w), dtype=np.float64)
        else:
            full = np.zeros((h, w), dtype=np.float64)
            count = np.zeros((h, w), dtype=np.float64)

        for y, x, patch in patches:
            patch = patch.astype(np.float64)

            if patch.ndim == 2:
                ph, pw = patch.shape
                _apply_merge_2d(full, count, patch, y, x, ph, pw, mode)
            elif patch.ndim == 3:
                _, ph, pw = patch.shape
                _apply_merge_3d(full, count, patch, y, x, ph, pw, mode)

        # Normaliser si mode mean
        if mode == "mean":
            count[count == 0] = 1
            full = full / count

        merged[base_name] = full

    # Ajouter les images non-groupées
    merged.update(ungrouped)

    return merged


def _apply_merge_2d(
    full: np.ndarray,
    count: np.ndarray,
    patch: np.ndarray,
    y: int,
    x: int,
    ph: int,
    pw: int,
    mode: str,
) -> None:
    """Applique la fusion pour un patch 2D (HW)."""
    if mode == "overwrite":
        full[y : y + ph, x : x + pw] = patch
    elif mode == "mean":
        full[y : y + ph, x : x + pw] += patch
        count[y : y + ph, x : x + pw] += 1
    elif mode == "max":
        sub = full[y : y + ph, x : x + pw]
        full[y : y + ph, x : x + pw] = np.maximum(sub, patch)
    elif mode == "min":
        sub = full[y : y + ph, x : x + pw]
        mask = count[y : y + ph, x : x + pw] == 0
        sub[mask] = patch[mask]
        full[y : y + ph, x : x + pw] = np.minimum(sub, patch)
        count[y : y + ph, x : x + pw] += 1
    else:
        raise ValueError(f"Mode de fusion inconnu : {mode}")


def _apply_merge_3d(
    full: np.ndarray,
    count: np.ndarray,
    patch: np.ndarray,
    y: int,
    x: int,
    ph: int,
    pw: int,
    mode: str,
) -> None:
    """Applique la fusion pour un patch 3D (CHW)."""
    if mode == "overwrite":
        full[:, y : y + ph, x : x + pw] = patch
    elif mode == "mean":
        full[:, y : y + ph, x : x + pw] += patch
        count[:, y : y + ph, x : x + pw] += 1
    elif mode == "max":
        sub = full[:, y : y + ph, x : x + pw]
        full[:, y : y + ph, x : x + pw] = np.maximum(sub, patch)
    elif mode == "min":
        sub = full[:, y : y + ph, x : x + pw]
        mask = count[:, y : y + ph, x : x + pw] == 0
        sub[mask] = patch[mask]
        full[:, y : y + ph, x : x + pw] = np.minimum(sub, patch)
        count[:, y : y + ph, x : x + pw] += 1
    else:
        raise ValueError(f"Mode de fusion inconnu : {mode}")


def merge_batch_predictions(
    names: List[str],
    pred_masks: torch.Tensor,
    origin_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    sw_metas: Optional[List[Dict]] = None,
    mode: str = "mean",
) -> Dict[str, np.ndarray]:
    """Fusionne les prédictions d'un batch de patches en images complètes.

    Fonction utilitaire pour l'inférence : prend directement les sorties
    du dataloader et les fusionne.

    Args:
        names: Liste des noms de patches (depuis data_dict['name']).
        pred_masks: Tensor de prédictions (B, C, H, W) ou (B, H, W).
        origin_sizes: Dictionnaire optionnel {nom_base: (H, W)}.
            Si non fourni, les tailles sont inférées depuis sw_metas.
        sw_metas: Liste optionnelle des métadonnées sliding window
            (depuis data_dict['sw_meta']).
        mode: Mode de fusion.

    Returns:
        Dictionnaire {nom_base: prediction_reconstruite}.
    """
    # Construire origin_sizes depuis sw_metas si disponible
    if origin_sizes is None and sw_metas is not None:
        origin_sizes = {}
        for meta in sw_metas:
            if meta.get("split", False):
                origin_sizes[meta["original_name"]] = (
                    meta["origin_h"],
                    meta["origin_w"],
                )

    if origin_sizes is None:
        origin_sizes = {}

    # Convertir le tensor en dictionnaire de numpy arrays
    predictions = {}
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()

    for i, name in enumerate(names):
        predictions[name] = pred_masks[i]

    return merge_predictions(predictions, origin_sizes, mode=mode)
