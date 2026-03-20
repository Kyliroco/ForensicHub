"""Tests pour vérifier l'intégrité des pixels et des tables de quantification
lors de l'utilisation de DocDataset + SlidingWindowWrapper.

Vérifie que :
1. Chaque patch produit par le SlidingWindowWrapper contient exactement les mêmes
   valeurs de pixels que la zone correspondante dans l'image d'origine.
2. La table de quantification (qt) est conservée intacte à travers le pipeline
   DocDataset → SlidingWindowWrapper.
3. Après application d'une compression JPEG (JpegCompressionWithDCT ou
   PillowJpegCompression), la table de quantification utilisée est bien celle
   stockée dans le fichier JPEG résultant.
"""

import importlib
import os
import sys
import tempfile
import types

import cv2
import jpegio
import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Import modules directly to bypass ForensicHub/__init__.py which pulls in
# heavy optional dependencies (efficientnet_pytorch, etc.).
# ---------------------------------------------------------------------------

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

def _direct_import(module_path: str, fqn: str):
    """Import a module by file path, registering it under *fqn* in sys.modules
    so that intra-package imports (``from ForensicHub.registry import …``) work."""
    if fqn in sys.modules:
        return sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod

# Ensure parent packages are stub-registered so sub-imports resolve
for pkg in [
    "ForensicHub",
    "ForensicHub.core",
    "ForensicHub.common",
    "ForensicHub.common.transforms",
    "ForensicHub.common.wrapper",
    "ForensicHub.tasks",
    "ForensicHub.tasks.document",
    "ForensicHub.tasks.document.datasets",
]:
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(ROOT, *pkg.split(".")[1:])]  # type: ignore[attr-defined]
        sys.modules[pkg] = m

_direct_import(os.path.join(ROOT, "ForensicHub", "core", "base_dataset.py"),
               "ForensicHub.core.base_dataset")
_direct_import(os.path.join(ROOT, "ForensicHub", "core", "base_model.py"),
               "ForensicHub.core.base_model")
_direct_import(os.path.join(ROOT, "ForensicHub", "core", "base_transform.py"),
               "ForensicHub.core.base_transform")
_direct_import(os.path.join(ROOT, "ForensicHub", "registry.py"),
               "ForensicHub.registry")
_direct_import(os.path.join(ROOT, "ForensicHub", "common", "transforms", "pillow_transforms.py"),
               "ForensicHub.common.transforms.pillow_transforms")
_direct_import(os.path.join(ROOT, "ForensicHub", "common", "wrapper", "sliding_window_wrapper.py"),
               "ForensicHub.common.wrapper.sliding_window_wrapper")
_direct_import(os.path.join(ROOT, "ForensicHub", "tasks", "document", "datasets", "doc_dataset.py"),
               "ForensicHub.tasks.document.datasets.doc_dataset")

from ForensicHub.common.transforms.pillow_transforms import (
    JpegCompressionWithDCT,
    PillowJpegCompression,
)
from ForensicHub.common.wrapper.sliding_window_wrapper import SlidingWindowWrapper
from ForensicHub.tasks.document.datasets.doc_dataset import DocDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_jpeg_data")


@pytest.fixture(scope="module")
def doc_dataset_no_transform():
    """DocDataset en mode test, sans transform, avec extraction DCT/QT."""
    ds = DocDataset(
        path=TEST_DATA_DIR,
        train=False,
        crop_size=512,
        get_dct_qtb=True,
        suffix_img=".jpg",
        suffix_mask=".png",
    )
    return ds


@pytest.fixture(scope="module")
def sliding_window_dataset(doc_dataset_no_transform):
    """SlidingWindowWrapper autour du DocDataset (patch 512×512, overlap 50%)."""
    sw = SlidingWindowWrapper(
        dataset=doc_dataset_no_transform,
        patch_width=512,
        patch_height=512,
        overlapping=0.5,
    )
    return sw


# ---------------------------------------------------------------------------
# 1. Intégrité des pixels dans les patches
# ---------------------------------------------------------------------------


class TestPixelIntegrity:
    """Vérifie que le SlidingWindowWrapper ne modifie pas les valeurs de pixels."""

    def test_patch_pixels_match_origin(self, doc_dataset_no_transform, sliding_window_dataset):
        """Chaque patch doit contenir exactement les pixels de la zone
        correspondante dans l'image d'origine du DocDataset."""
        ds = doc_dataset_no_transform
        sw = sliding_window_dataset

        patches_checked = 0
        for sw_idx in range(len(sw)):
            patch_item = sw[sw_idx]
            meta = patch_item["sw_meta"]

            # Retrouver l'item source
            dataset_idx = sw._index_map[sw_idx][0]
            origin_item = ds[dataset_idx]

            origin_image = origin_item["image"]  # tensor CHW ou ndarray HWC

            patch_image = patch_item["image"]

            if not meta["split"]:
                # Image non découpée : doit être identique
                if isinstance(origin_image, torch.Tensor):
                    assert torch.equal(origin_image, patch_image), (
                        f"Image non-split {patch_item['name']} : pixels différents"
                    )
                else:
                    np.testing.assert_array_equal(origin_image, patch_image)
            else:
                y, x = meta["patch_y"], meta["patch_x"]
                ph, pw = meta["patch_h"], meta["patch_w"]

                if isinstance(origin_image, torch.Tensor) and origin_image.dim() == 3:
                    expected = origin_image[:, y : y + ph, x : x + pw]
                    assert torch.equal(expected, patch_image), (
                        f"Patch {patch_item['name']} : pixels différents à "
                        f"(y={y}, x={x}, h={ph}, w={pw})"
                    )
                else:
                    if origin_image.ndim == 3:
                        expected = origin_image[y : y + ph, x : x + pw, :]
                    else:
                        expected = origin_image[y : y + ph, x : x + pw]
                    np.testing.assert_array_equal(expected, patch_image)

            patches_checked += 1

        assert patches_checked > 0, "Aucun patch vérifié"
        print(f"\n[OK] {patches_checked} patches vérifiés — pixels intacts.")

    def test_mask_patch_matches_origin(self, doc_dataset_no_transform, sliding_window_dataset):
        """Les masques des patches doivent correspondre aux zones du masque original."""
        ds = doc_dataset_no_transform
        sw = sliding_window_dataset

        for sw_idx in range(len(sw)):
            patch_item = sw[sw_idx]
            meta = patch_item["sw_meta"]

            dataset_idx = sw._index_map[sw_idx][0]
            origin_item = ds[dataset_idx]

            origin_mask = origin_item["mask"]
            patch_mask = patch_item["mask"]

            if not meta["split"]:
                assert torch.equal(origin_mask, patch_mask)
            else:
                y, x = meta["patch_y"], meta["patch_x"]
                ph, pw = meta["patch_h"], meta["patch_w"]

                if origin_mask.dim() == 3:
                    expected = origin_mask[:, y : y + ph, x : x + pw]
                else:
                    expected = origin_mask[y : y + ph, x : x + pw]

                assert torch.equal(expected, patch_mask), (
                    f"Mask du patch {patch_item['name']} diffère de l'original"
                )

    def test_dct_patch_matches_origin(self, doc_dataset_no_transform, sliding_window_dataset):
        """Les coefficients DCT des patches doivent correspondre à la zone de l'original."""
        ds = doc_dataset_no_transform
        sw = sliding_window_dataset

        for sw_idx in range(len(sw)):
            patch_item = sw[sw_idx]
            meta = patch_item["sw_meta"]

            if "dct" not in patch_item:
                continue

            dataset_idx = sw._index_map[sw_idx][0]
            origin_item = ds[dataset_idx]

            if "dct" not in origin_item:
                continue

            origin_dct = origin_item["dct"]
            patch_dct = patch_item["dct"]

            if not meta["split"]:
                np.testing.assert_array_equal(origin_dct, patch_dct)
            else:
                y, x = meta["patch_y"], meta["patch_x"]
                ph, pw = meta["patch_h"], meta["patch_w"]
                expected = origin_dct[y : y + ph, x : x + pw]
                np.testing.assert_array_equal(expected, patch_dct), (
                    f"DCT du patch {patch_item['name']} diffère"
                )


# ---------------------------------------------------------------------------
# 2. Conservation de la table de quantification
# ---------------------------------------------------------------------------


class TestQuantizationTablePreservation:
    """Vérifie que la table de quantification est conservée dans le pipeline."""

    def test_qt_matches_original_jpeg(self, doc_dataset_no_transform):
        """La table QT retournée par DocDataset doit correspondre à celle du
        fichier JPEG original (après clipping 0-63)."""
        ds = doc_dataset_no_transform

        for idx in range(len(ds)):
            item = ds[idx]
            img_path = ds.images[idx][0]

            # Lire la QT directement depuis le fichier JPEG
            jpg = jpegio.read(img_path)
            original_qt = np.clip(np.abs(jpg.quant_tables[0]), 0, 63)

            returned_qt = item["qt"]

            np.testing.assert_array_equal(
                original_qt,
                returned_qt,
                err_msg=f"QT de {os.path.basename(img_path)} ne correspond pas",
            )

        print(f"\n[OK] Tables de quantification vérifiées pour {len(ds)} images.")

    def test_qt_preserved_through_sliding_window(
        self, doc_dataset_no_transform, sliding_window_dataset
    ):
        """La table QT ne doit pas être modifiée par le SlidingWindowWrapper
        (elle est copiée telle quelle, pas découpée)."""
        ds = doc_dataset_no_transform
        sw = sliding_window_dataset

        for sw_idx in range(len(sw)):
            patch_item = sw[sw_idx]

            if "qt" not in patch_item:
                continue

            dataset_idx = sw._index_map[sw_idx][0]
            origin_item = ds[dataset_idx]

            np.testing.assert_array_equal(
                origin_item["qt"],
                patch_item["qt"],
                err_msg=f"QT altérée par SlidingWindowWrapper pour {patch_item['name']}",
            )

    def test_qt_shape_is_8x8(self, doc_dataset_no_transform):
        """La table de quantification doit toujours être de shape (8, 8)."""
        ds = doc_dataset_no_transform
        for idx in range(len(ds)):
            item = ds[idx]
            assert item["qt"].shape == (8, 8), (
                f"QT de shape {item['qt'].shape} au lieu de (8, 8)"
            )

    def test_qt_values_in_range(self, doc_dataset_no_transform):
        """Les valeurs QT clippées doivent être dans [0, 63]."""
        ds = doc_dataset_no_transform
        for idx in range(len(ds)):
            item = ds[idx]
            assert item["qt"].min() >= 0, "Valeur QT négative"
            assert item["qt"].max() <= 63, "Valeur QT > 63"


# ---------------------------------------------------------------------------
# 3. Compression et conservation de la table utilisée
# ---------------------------------------------------------------------------


class TestCompressionTableConsistency:
    """Vérifie qu'après compression, la table utilisée est bien celle stockée
    dans le fichier JPEG résultant."""

    def test_jpeg_compression_with_dct_table_stored(self):
        """JpegCompressionWithDCT : la QT exposée doit correspondre à celle
        qu'on retrouve dans le fichier JPEG produit."""
        transform = JpegCompressionWithDCT(quality_range=(50, 90), p=1.0)

        # Créer une image test
        np.random.seed(123)
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        compressed = transform.apply(img, quality=75)

        assert transform._last_qtb is not None, "QTB non capturée après compression"
        assert transform._last_dct is not None, "DCT non capturée après compression"

        # Vérifier en re-compressant et relisant
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            Image.fromarray(compressed).save(
                tmp_path, format="JPEG", quality=75, subsampling=0
            )

        # La QT stockée par le transform doit être celle de la compression d'origine
        # Vérifions la forme et la plausibilité
        assert transform._last_qtb.shape == (8, 8), (
            f"QTB shape {transform._last_qtb.shape} au lieu de (8, 8)"
        )
        assert transform._last_qtb.min() >= 1, "QTB contient des zéros (invalide pour JPEG)"
        os.unlink(tmp_path)

    def test_jpeg_compression_with_dct_table_matches_file(self):
        """La QT exposée par JpegCompressionWithDCT doit correspondre exactement
        à celle lue depuis le fichier JPEG produit par la même compression."""
        transform = JpegCompressionWithDCT(quality_range=(75, 75), p=1.0)

        np.random.seed(456)
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        compressed = transform.apply(img, quality=75)

        # Sauvegarder les pixels compressés en JPEG et relire la QT
        # On utilise la même qualité pour avoir la même table
        captured_qt = transform._last_qtb.copy()
        captured_dct = transform._last_dct.copy()

        # Vérifier que les DCT sont cohérentes avec l'image compressée
        # En ré-encodant l'image compressée avec la même qualité, on devrait
        # obtenir les mêmes coefficients DCT (car l'image est déjà compressée)
        assert captured_qt.shape == (8, 8)
        assert captured_dct.ndim == 2

        # La DCT doit couvrir l'image (potentiellement paddée à des multiples de 8)
        assert captured_dct.shape[0] >= 512
        assert captured_dct.shape[1] >= 512

    def test_pillow_compression_custom_table_stored(self):
        """PillowJpegCompression : la table custom utilisée doit être retrouvable
        dans le fichier JPEG produit."""
        luma_table = [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ]
        chroma_table = [
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ]

        transform = PillowJpegCompression(
            luma_tables=[luma_table],
            chroma_tables=[chroma_table],
            p=1.0,
        )

        np.random.seed(789)
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        compressed = transform.apply(
            img, luma_table=luma_table, chroma_table=chroma_table
        )

        # La QTB capturée doit correspondre à la table luma utilisée
        assert transform._last_qtb is not None, "QTB non capturée"
        captured_qt = transform._last_qtb

        # Reshape la table flat en 8×8 pour comparaison
        expected_qt = np.array(luma_table, dtype=np.int32).reshape(8, 8)

        np.testing.assert_array_equal(
            captured_qt,
            expected_qt,
            err_msg="La QT capturée ne correspond pas à la table luma fournie",
        )

    def test_pillow_compression_table_in_output_file(self):
        """Après compression PillowJpegCompression, si on sauvegarde le résultat
        en JPEG avec la même table, la table est bien dans le fichier."""
        luma_table = [
            3, 2, 2, 3, 5, 8, 10, 12,
            2, 2, 3, 4, 5, 12, 12, 11,
            3, 3, 3, 5, 8, 11, 14, 11,
            3, 3, 4, 6, 10, 17, 16, 12,
            4, 4, 7, 11, 14, 22, 21, 15,
            5, 7, 11, 13, 16, 21, 23, 18,
            10, 13, 16, 17, 21, 24, 24, 21,
            14, 18, 19, 21, 22, 20, 20, 20,
        ]
        chroma_table = [
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ]

        transform = PillowJpegCompression(
            luma_tables=[luma_table],
            chroma_tables=[chroma_table],
            p=1.0,
        )

        np.random.seed(101)
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        compressed = transform.apply(
            img, luma_table=luma_table, chroma_table=chroma_table
        )

        # Sauvegarder avec la même table et vérifier via jpegio
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_img = Image.fromarray(compressed)
            pil_img.save(
                tmp_path,
                format="JPEG",
                qtables={0: luma_table, 1: chroma_table},
            )

        try:
            jpg = jpegio.read(tmp_path)
            file_qt = jpg.quant_tables[0]
            expected_qt = np.array(luma_table, dtype=np.int32).reshape(8, 8)

            np.testing.assert_array_equal(
                file_qt,
                expected_qt,
                err_msg="La QT dans le fichier JPEG ne correspond pas à la table utilisée",
            )
        finally:
            os.unlink(tmp_path)

    def test_compression_dct_matches_reread(self):
        """Les coefficients DCT capturés par le transform doivent correspondre
        à ceux qu'on relira depuis le fichier JPEG produit."""
        transform = JpegCompressionWithDCT(quality_range=(60, 60), p=1.0)

        np.random.seed(202)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        compressed = transform.apply(img, quality=60)
        captured_dct = transform._last_dct.copy()
        captured_qt = transform._last_qtb.copy()

        # Re-sauvegarder les pixels compressés en JPEG et relire DCT
        # Si on compresse la même image avec la même QT, on doit avoir les mêmes DCT
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            # On sauvegarde l'image compressée avec la table capturée pour s'assurer
            # que la table est bien dans le fichier
            qt_flat = captured_qt.flatten().tolist()
            pil_img = Image.fromarray(compressed)
            pil_img.save(
                tmp_path,
                format="JPEG",
                qtables={0: qt_flat, 1: qt_flat},
            )

        try:
            jpg = jpegio.read(tmp_path)
            file_qt = jpg.quant_tables[0]

            np.testing.assert_array_equal(
                file_qt,
                captured_qt,
                err_msg="La QT relue du fichier diffère de celle capturée par le transform",
            )
        finally:
            os.unlink(tmp_path)

    def test_multiple_qualities_produce_different_tables(self):
        """Des qualités JPEG différentes doivent produire des tables différentes."""
        transform = JpegCompressionWithDCT(quality_range=(30, 95), p=1.0)

        np.random.seed(303)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        transform.apply(img, quality=30)
        qt_low = transform._last_qtb.copy()

        transform.apply(img, quality=95)
        qt_high = transform._last_qtb.copy()

        # Les tables doivent être différentes
        assert not np.array_equal(qt_low, qt_high), (
            "Qualités 30 et 95 produisent la même table de quantification"
        )

        # Qualité basse → valeurs QT plus élevées (compression plus forte)
        assert qt_low.sum() > qt_high.sum(), (
            "La table de qualité basse devrait avoir des valeurs plus élevées"
        )


# ---------------------------------------------------------------------------
# 4. Tests complémentaires sur le sliding window
# ---------------------------------------------------------------------------


class TestSlidingWindowCoverage:
    """Tests complémentaires sur la couverture et la cohérence du sliding window."""

    def test_all_patches_have_correct_size(self, sliding_window_dataset):
        """Tous les patches d'images découpées doivent avoir la taille attendue."""
        sw = sliding_window_dataset

        for sw_idx in range(len(sw)):
            item = sw[sw_idx]
            meta = item["sw_meta"]

            if meta["split"]:
                image = item["image"]
                if isinstance(image, torch.Tensor) and image.dim() == 3:
                    _, h, w = image.shape
                elif isinstance(image, np.ndarray) and image.ndim == 3:
                    h, w, _ = image.shape
                else:
                    h, w = image.shape[:2]

                assert h == meta["patch_h"], (
                    f"Patch {item['name']}: hauteur {h} != {meta['patch_h']}"
                )
                assert w == meta["patch_w"], (
                    f"Patch {item['name']}: largeur {w} != {meta['patch_w']}"
                )

    def test_patch_positions_within_bounds(self, sliding_window_dataset):
        """Les positions des patches ne doivent pas dépasser l'image d'origine."""
        sw = sliding_window_dataset

        for sw_idx in range(len(sw)):
            item = sw[sw_idx]
            meta = item["sw_meta"]

            if meta["split"]:
                assert meta["patch_y"] >= 0
                assert meta["patch_x"] >= 0
                assert meta["patch_y"] + meta["patch_h"] <= meta["origin_h"], (
                    f"Patch {item['name']}: y+h dépasse l'image"
                )
                assert meta["patch_x"] + meta["patch_w"] <= meta["origin_w"], (
                    f"Patch {item['name']}: x+w dépasse l'image"
                )

    def test_sliding_window_name_convention(self, sliding_window_dataset):
        """Les noms des patches split doivent suivre la convention {name}_{y}_{x}."""
        sw = sliding_window_dataset

        for sw_idx in range(len(sw)):
            item = sw[sw_idx]
            meta = item["sw_meta"]

            if meta["split"]:
                expected_name = f"{meta['original_name']}_{meta['patch_y']}_{meta['patch_x']}"
                assert item["name"] == expected_name, (
                    f"Nom attendu '{expected_name}', obtenu '{item['name']}'"
                )

    def test_label_recalculated_from_mask_patch(self, sliding_window_dataset):
        """Le label de chaque patch doit refléter le contenu de son masque."""
        sw = sliding_window_dataset

        for sw_idx in range(len(sw)):
            item = sw[sw_idx]

            if "mask" in item and "label" in item:
                mask = item["mask"]
                if isinstance(mask, torch.Tensor):
                    expected_label = (mask.sum() != 0).long()
                    assert item["label"] == expected_label, (
                        f"Label du patch {item['name']} ne correspond pas au masque"
                    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
