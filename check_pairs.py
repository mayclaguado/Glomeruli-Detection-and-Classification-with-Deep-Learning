#!/usr/bin/env python3
# check_pairs_paths.py
# Pega tus rutas aquí 👇
IMAGES_DIR = r"G:\dataset_parcial_prueba_para_winner_15\images"
LABELS_DIR = r"G:\dataset_parcial_prueba_para_winner_15\labels"

from pathlib import Path
from collections import defaultdict
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LBL_EXT  = ".txt"

def collect_stems(dir_path: Path, kind: str):
    stems = defaultdict(list)  # stem -> [full_paths]
    for p in dir_path.rglob("*"):
        if p.is_file():
            if kind == "img" and p.suffix.lower() in IMG_EXTS:
                stems[p.stem].append(p)
            if kind == "lbl" and p.suffix.lower() == LBL_EXT:
                stems[p.stem].append(p)
    return stems

def main(images_dir: Path, labels_dir: Path):
    if not images_dir.exists() or not labels_dir.exists():
        sys.exit(f"Ruta no válida.\n  IMAGES_DIR: {images_dir}\n  LABELS_DIR: {labels_dir}")

    img = collect_stems(images_dir, "img")
    lbl = collect_stems(labels_dir, "lbl")

    img_stems = set(img.keys())
    lbl_stems = set(lbl.keys())

    missing_labels = sorted(img_stems - lbl_stems)   # hay imagen pero falta .txt
    missing_images = sorted(lbl_stems - img_stems)   # hay .txt pero falta imagen

    # Duplicados: mismo stem con varias extensiones/archivos
    img_dups = {k:v for k,v in img.items() if len(v) > 1}
    lbl_dups = {k:v for k,v in lbl.items() if len(v) > 1}

    # Reporte
    print("=== RESUMEN ===")
    print(f"Carpeta imágenes: {images_dir}")
    print(f"Carpeta labels  : {labels_dir}")
    print(f"Total imágenes: {sum(len(v) for v in img.values())} (stems únicos: {len(img_stems)})")
    print(f"Total labels  : {sum(len(v) for v in lbl.values())} (stems únicos: {len(lbl_stems)})")
    print(f"Imágenes sin label: {len(missing_labels)}")
    print(f"Labels sin imagen: {len(missing_images)}")
    print(f"Stems con múltiples IMÁGENES: {len(img_dups)}")
    print(f"Stems con múltiples LABELS : {len(lbl_dups)}")
    print()

    if missing_labels:
        print(">> Imágenes SIN label (.txt):")
        for s in missing_labels:
            print("  -", ", ".join(str(p) for p in img[s]))
        print()

    if missing_images:
        print(">> Labels SIN imagen:")
        for s in missing_images:
            print("  -", ", ".join(str(p) for p in lbl[s]))
        print()

    if img_dups:
        print(">> Stems con varias imágenes (posible conflicto de extensiones):")
        for s, paths in img_dups.items():
            print("  -", s, "->", ", ".join(p.suffix for p in paths), "|", ", ".join(str(p) for p in paths))
        print()

    if lbl_dups:
        print(">> Stems con varios .txt (debería ser 1):")
        for s, paths in lbl_dups.items():
            print("  -", s, "->", ", ".join(str(p) for p in paths))

if __name__ == "__main__":
    images = Path(IMAGES_DIR)
    labels = Path(LABELS_DIR)
    main(images, labels)


