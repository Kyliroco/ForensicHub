#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG", ".png"}
MSK_EXTS = {".png", ".PNG"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    percent: float  # 0-100


def list_files_flat(dirpath: Path, exts: set[str]) -> Dict[str, Path]:
    """Map stem -> file path (non-recursive)."""
    out: Dict[str, Path] = {}
    if not dirpath.is_dir():
        return out
    for p in dirpath.iterdir():
        if p.is_file() and p.suffix in exts:
            out[p.stem] = p
    return out


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        dst.unlink()
    os.symlink(src, dst)


def parse_config(cfg_path: Path) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) != 3:
            raise ValueError(f"Invalid config line (expected name;root;percent): {line}")
        name, root_str, pct_str = parts
        root = Path(root_str).expanduser()
        pct = float(pct_str)
        if not (0 <= pct <= 100):
            raise ValueError(f"Percent must be in [0,100], got {pct} for {name}")
        specs.append(DatasetSpec(name=name, root=root, percent=pct))
    if not specs:
        raise ValueError("Config is empty.")
    return specs


def safe_rmtree(path: Path, *, dry_run: bool) -> None:
    """
    Remove an output directory safely.
    Refuses to delete root, empty paths, or non-directories.
    """
    p = path.expanduser()

    # Basic safety checks
    if not str(p).strip():
        raise ValueError("Refusing to delete empty path.")
    if p.resolve() == Path("/").resolve():
        raise ValueError("Refusing to delete '/'.")
    if not p.exists():
        return
    if not p.is_dir():
        raise ValueError(f"Refusing to delete non-directory path: {p}")

    if dry_run:
        print(f"[dry-run] Would remove output directory: {p}")
        return

    shutil.rmtree(p)
    print(f"Removed output directory: {p}")


def link_one(
    out_root: Path,
    ds_name: str,
    stem: str,
    imgs: Dict[str, Path],
    msks: Dict[str, Path],
    dry_run: bool,
    train: bool = False,
) -> int:
    """Create symlinks for one item. Returns number of links created."""
    created = 0

    img_path = imgs[stem]
    if train:
        out_img = out_root / "images" / (str(ds_name) + "_" + str(img_path.name))
    else:
        out_img = out_root / ds_name / "images" / img_path.name

    if dry_run:
        print(f"  IMG {out_img} -> {img_path}")
    else:
        safe_symlink(img_path, out_img)
        created += 1

    msk_path: Optional[Path] = msks.get(stem)
    if msk_path is not None:
        if train:
            out_msk = out_root / "masks" / (str(ds_name) + "_" + str(msk_path.name))
        else:
            out_msk = out_root / ds_name / "masks" / msk_path.name

        if dry_run:
            print(f"  MSK {out_msk} -> {msk_path}")
        else:
            safe_symlink(msk_path, out_msk)
            created += 1

    return created


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample dataset items and create symlinks; optionally also output the complementary 'rest' split."
    )
    ap.add_argument("--config", required=True, type=Path, help="Config file: name;root;percent")
    ap.add_argument("--out", required=True, type=Path, help="Output folder for selected symlinks")
    ap.add_argument(
        "--out-rest",
        type=Path,
        default=None,
        help="Optional output folder for the complementary split (everything NOT selected).",
    )
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--min-keep", type=int, default=1, help="Min items kept if percent>0 and dataset non-empty")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without linking")
    ap.add_argument(
        "--require-masks",
        action="store_true",
        help="If set: only sample items that have both image+mask (pair-only mode).",
    )
    args = ap.parse_args()

    # --- NEW: clean output dirs before starting ---
    safe_rmtree(args.out, dry_run=args.dry_run)
    if args.out_rest is not None:
        safe_rmtree(args.out_rest, dry_run=args.dry_run)

    rng = random.Random(args.seed)
    specs = parse_config(args.config)

    total_links_sel = 0
    total_links_rest = 0

    for spec in specs:
        images_dir = spec.root / "images"
        masks_dir = spec.root / "masks"

        if not images_dir.is_dir():
            print(f"[{spec.name}] Missing images/ folder, skipping: {images_dir}")
            continue

        imgs = list_files_flat(images_dir, IMG_EXTS)
        msks = list_files_flat(masks_dir, MSK_EXTS) if masks_dir.is_dir() else {}

        if not imgs:
            print(f"[{spec.name}] 0 images found, skipping.")
            continue

        if args.require_masks:
            keys = sorted(set(imgs.keys()) & set(msks.keys()))
            mode = "pair-only"
        else:
            keys = sorted(imgs.keys())
            mode = "image-first"

        if not keys:
            print(f"[{spec.name}] mode={mode} -> 0 eligible items. Skipping.")
            continue

        n = len(keys)
        k = int(math.floor(n * (spec.percent / 100.0)))
        if k == 0 and spec.percent > 0:
            k = max(args.min_keep, 1)
        k = min(k, n)

        rng.shuffle(keys)
        chosen = keys[:k]
        rest = keys[k:]

        print(
            f"[{spec.name}] mode={mode} images={len(imgs)} masks_folder={masks_dir.is_dir()} "
            f"eligible={n} keep={len(chosen)} rest={len(rest)}"
        )

        # Selected split
        for stem in chosen:
            total_links_sel += link_one(args.out, spec.name, stem, imgs, msks, args.dry_run, train=True)

        # Rest split (optional)
        if args.out_rest is not None:
            for stem in rest:
                total_links_rest += link_one(args.out_rest, spec.name, stem, imgs, msks, args.dry_run)

    if not args.dry_run:
        print(f"Done. Selected links: {total_links_sel} in {args.out}")
        if args.out_rest is not None:
            print(f"Done. Rest links: {total_links_rest} in {args.out_rest}")


if __name__ == "__main__":
    main()
