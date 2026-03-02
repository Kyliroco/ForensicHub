#!/usr/bin/env python3
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    max_total: int  # max pool size (train+test)
    train_percent: float  # 0-100
    test_percent: float   # 0-100
import re

TILE_RE = re.compile(r"^(?P<base>.+)_(?P<h>\d+)_(?P<w>\d+)$")

def root_stem(stem: str) -> str:
    """
    Return the base stem without trailing _H_W if present.
    """
    m = TILE_RE.match(stem)
    if m:
        return m.group("base")
    return stem


def group_by_root(stems: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for s in stems:
        groups.setdefault(root_stem(s), []).append(s)
    return groups

def has_origin_tile(group: List[str]) -> bool:
    """
    Rules:
    - If group has only one file -> always valid (standalone image)
    - If group has multiple tiles -> must contain a _0_0 tile
    """
    if len(group) == 1:
        return True

    for stem in group:
        if stem.endswith("_0_0"):
            return True

    return False


def list_files_flat(dirpath: Path, exts: set[str]) -> Dict[str, Path]:
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


def safe_rmtree(path: Path, *, dry_run: bool) -> None:
    p = path.expanduser()
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


def parse_config(cfg_path: Path) -> List[DatasetSpec]:
    """
    Format per line:
      name;root;max_total;train_percent;test_percent

    Example:
      ds1;/data/ds1;100;70;20
    """
    specs: List[DatasetSpec] = []
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) != 5:
            raise ValueError(
                f"Invalid config line (expected name;root;max_total;train_percent;test_percent): {line}"
            )
        name, root_str, max_str, train_str, test_str = parts
        root = Path(root_str).expanduser()

        max_total = int(max_str)
        train_pct = float(train_str)
        test_pct = float(test_str)

        if max_total < 0:
            raise ValueError(f"max_total must be >= 0, got {max_total} for {name}")
        if not (0 <= train_pct <= 100) or not (0 <= test_pct <= 100):
            raise ValueError(f"Percents must be in [0,100], got train={train_pct}, test={test_pct} for {name}")
        if train_pct + test_pct > 100:
            raise ValueError(f"train_percent + test_percent must be <= 100, got {train_pct + test_pct} for {name}")

        specs.append(
            DatasetSpec(
                name=name,
                root=root,
                max_total=max_total,
                train_percent=train_pct,
                test_percent=test_pct,
            )
        )

    if not specs:
        raise ValueError("Config is empty.")
    return specs


def link_one(
    out_root: Path,
    ds_name: str,
    stem: str,
    imgs: Dict[str, Path],
    msks: Dict[str, Path],
    dry_run: bool,
    train_flat: bool = False,
) -> int:
    created = 0

    img_path = imgs[stem]
    if train_flat:
        out_img = out_root / "images" / f"{ds_name}_{img_path.name}"
    else:
        out_img = out_root / ds_name / "images" / img_path.name

    if dry_run:
        print(f"  IMG {out_img} -> {img_path}")
    else:
        safe_symlink(img_path, out_img)
        created += 1

    msk_path: Optional[Path] = msks.get(stem)
    if msk_path is not None:
        if train_flat:
            out_msk = out_root / "masks" / f"{ds_name}_{msk_path.name}"
        else:
            out_msk = out_root / ds_name / "masks" / msk_path.name

        if dry_run:
            print(f"  MSK {out_msk} -> {msk_path}")
        else:
            safe_symlink(msk_path, out_msk)
            created += 1

    return created

def process_dataset(spec: DatasetSpec, args, rng_seed: int):
    rng = random.Random(rng_seed)

    total_links_train = 0
    total_links_test = 0

    images_dir = spec.root / "images"
    masks_dir = spec.root / "masks"

    if not images_dir.is_dir():
        print(f"[{spec.name}] Missing images/ folder, skipping: {images_dir}")
        return 0, 0

    imgs = list_files_flat(images_dir, IMG_EXTS)
    msks = list_files_flat(masks_dir, MSK_EXTS) if masks_dir.is_dir() else {}

    if not imgs:
        print(f"[{spec.name}] 0 images found, skipping.")
        return 0, 0

    img_groups_all = group_by_root(list(imgs.keys()))
    msk_groups_all = group_by_root(list(msks.keys())) if msks else {}

    img_groups = {k: v for k, v in img_groups_all.items() if has_origin_tile(v)}
    msk_groups = {k: v for k, v in msk_groups_all.items() if has_origin_tile(v)} if msks else {}

    if args.require_masks:
        keys = sorted([k for k in img_groups.keys() if k in msk_groups])
        mode = "pair-only-group"
    else:
        keys = sorted(img_groups.keys())
        mode = "image-first-group"

    if not keys:
        print(f"[{spec.name}] mode={mode} -> 0 eligible items. Skipping.")
        return 0, 0

    pool = keys[:]
    rng.shuffle(pool)
    n_pool = len(pool)

    k_train = int(math.floor(n_pool * (spec.train_percent / 100.0)))
    k_test = int(math.floor(n_pool * (spec.test_percent / 100.0)))

    k_train = min(k_train, n_pool)
    k_test = min(k_test, n_pool - k_train)

    train_keys = pool[:k_train]
    test_keys = pool[k_train : k_train + k_test]
    ignored = pool[k_train + k_test :]

    len_train = 0
    for group in train_keys:
        len_train += len(img_groups[group])
        if len_train > spec.max_total:
            len_train -= len(img_groups[group])
            break
        for stem in img_groups[group]:
            total_links_train += link_one(
                args.out_train, spec.name, stem, imgs, msks, args.dry_run, train_flat=True
            )
    len_test = 0
    for group in test_keys:
        len_test += len(img_groups[group])
        if len_test > spec.max_total:
            len_test -= len(img_groups[group])
            break
        for stem in img_groups[group]:
            total_links_test += link_one(
                args.out_test, spec.name, stem, imgs, msks, args.dry_run, train_flat=False
            )
    print(
        f"[{spec.name}] mode={mode} eligible={len(keys)} "
        f"pool={n_pool} (max={spec.max_total}) "
        f"train={len_train} test={len_test} ignored={len(ignored)}"
    )  
    return total_links_train, total_links_test

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample per-dataset pool size + train/test percents from config and create symlinks."
    )
    ap.add_argument("--config", required=True, type=Path, help="Config: name;root;max_total;train_percent;test_percent")
    ap.add_argument("--out-train", required=True, type=Path, help="Output folder for TRAIN symlinks")
    ap.add_argument("--out-test", required=True, type=Path, help="Output folder for TEST symlinks")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without linking")
    ap.add_argument(
        "--require-masks",
        action="store_true",
        help="If set: only sample items that have both image+mask (pair-only mode).",
    )
    ap.add_argument(
        "--train-flat",
        action="store_true",
        help="If set: put train in out_train/images + out_train/masks with dsname_ prefix (like your previous train=True behavior).",
    )
    args = ap.parse_args()

    # Clean outputs
    safe_rmtree(args.out_train, dry_run=args.dry_run)
    safe_rmtree(args.out_test, dry_run=args.dry_run)

    rng = random.Random(args.seed)
    specs = parse_config(args.config)

    total_links_train = 0
    total_links_test = 0

    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = [
            executor.submit(process_dataset, spec, args, args.seed + i)
            for i, spec in enumerate(specs)
        ]

        for fut in as_completed(futures):
            train_links, test_links = fut.result()
            total_links_train += train_links
            total_links_test += test_links


    if not args.dry_run:
        print(f"Done. Train links: {total_links_train} in {args.out_train}")
        print(f"Done. Test links:  {total_links_test} in {args.out_test}")


if __name__ == "__main__":
    main()
