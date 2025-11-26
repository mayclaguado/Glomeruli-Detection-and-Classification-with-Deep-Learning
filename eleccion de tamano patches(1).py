import os, math, csv
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import box
from tqdm import tqdm


import openslide


# TIAToolbox (opcional)
try:
    from tiatoolbox.wsicore.wsireader import WSIReader
    from tiatoolbox.tools.patchextraction import PatchExtractor
    TIATOOLBOX_OK = True
except Exception:
    TIATOOLBOX_OK = False


# =========================
# Utilidades de anotaciones
# =========================


def load_bboxes_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    for c in ["x1","y1","x2","y2"]:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en {csv_path}")
    return df[["x1","y1","x2","y2"]].values.astype(float)


def load_bboxes_yolo_txt(txt_path: str, w: int, h: int) -> np.ndarray:
    bxs = []
    with open(txt_path, "r") as f:
        for ln in f:
            if not ln.strip(): continue
            parts = ln.strip().split()
            if len(parts) not in (5,6):  # "cls cx cy w h" o "cls conf cx cy w h"
                continue
            if len(parts) == 5:
                _, cx, cy, ww, hh = parts
            else:
                _, _, cx, cy, ww, hh = parts
            cx, cy, ww, hh = map(float, (cx, cy, ww, hh))
            cx *= w; cy *= h; ww *= w; hh *= h
            x1, y1 = cx - ww/2.0, cy - hh/2.0
            x2, y2 = cx + ww/2.0, cy + hh/2.0
            bxs.append([x1,y1,x2,y2])
    return np.array(bxs, dtype=float)


# =====================
# Metadatos de OpenSlide
# =====================


def get_wsi_props(slide: openslide.OpenSlide):
    props = slide.properties
    def _f(key):
        try: return float(props.get(key, ""))
        except: return None
    return {
        "mpp_x": _f("openslide.mpp-x"),
        "mpp_y": _f("openslide.mpp-y"),
        "objective_power": _f("openslide.objective-power")
    }


def scale_factor_for_mag(desired_mag: float,
                         base_mag: Optional[float],
                         mpp_x: Optional[float],
                         mpp_y: Optional[float]) -> float:
    """
    Factor 's' para pasar de coords nivel0 (resolución base) a desired_mag.
    Si hay mpp usamos relación de magnificación; si no, usamos base_mag/des_mag.
    """
    if mpp_x and mpp_y:
        # Asumimos relación lineal entre mag aparente y µm/px
        if base_mag is None:
            base_mag = 40.0  # fallback típico si no hay metadata
        base_mpp = (mpp_x + mpp_y) / 2.0
        target_mpp = base_mpp * (base_mag / desired_mag)
        return target_mpp / base_mpp  # == base_mag/desired_mag
    if base_mag:
        return float(base_mag / desired_mag)
    return float(40.0 / desired_mag)


def nearest_level_for_scale(slide: openslide.OpenSlide, s: float) -> int:
    downs = slide.level_downsamples
    diffs = [abs(d - s) for d in downs]
    return int(np.argmin(diffs))


# ============================
# Métricas de cobertura (fit %)
# ============================


def coverage_grid(bboxes_level0: np.ndarray,
                  w0: int, h0: int,
                  patch_size: int,
                  s: float,
                  overlap: float) -> float:
    """
    % de bboxes completamente contenidos en algún patch de REJILLA (con solape).
    patch_size: en píxeles a la magnificación deseada.
    s: factor nivel0 -> mag deseada (p.ej., 2.0 si 40×->20×).
    """
    P0 = patch_size * s
    stride0 = max(1.0, P0 * (1.0 - overlap))


    nx = max(1, int(math.ceil((w0 - P0) / stride0)) + 1)
    ny = max(1, int(math.ceil((h0 - P0) / stride0)) + 1)
    xs = [min(int(round(i * stride0)), max(0, int(w0 - P0))) for i in range(nx)]
    ys = [min(int(round(j * stride0)), max(0, int(h0 - P0))) for j in range(ny)]


    covered = 0
    for (x1,y1,x2,y2) in bboxes_level0:
        ok = False
        i_min = max(0, int((x1 - P0) // stride0))
        i_max = min(nx-1, int(x2 // stride0)+1)
        j_min = max(0, int((y1 - P0) // stride0))
        j_max = min(ny-1, int((y2 // stride0)+1))
        for i in range(i_min, i_max+1):
            if ok: break
            for j in range(j_min, j_max+1):
                rx1, ry1 = xs[i], ys[j]
                rx2, ry2 = rx1 + P0, ry1 + P0
                if rx1 <= x1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2:
                    ok = True
                    break
        covered += int(ok)
    total = max(1, len(bboxes_level0))
    return 100.0 * covered / total


def coverage_centered(bboxes_level0: np.ndarray,
                      patch_size: int,
                      s: float,
                      margin_frac: float = 0.10) -> float:
    """
    % de bboxes que caben si centramos un patch en el centro del bbox.
    Útil si tu extracción final será "un parche por glomérulo".
    """
    P0 = patch_size * s
    covered = 0
    for (x1,y1,x2,y2) in bboxes_level0:
        bw, bh = (x2-x1), (y2-y1)
        need = (1.0 + 2*margin_frac) * max(bw, bh)
        covered += int(P0 >= need)
    total = max(1, len(bboxes_level0))
    return 100.0 * covered / total


# =========================
# Probing y visualizaciones
# =========================


def run_probe(wsi_path: str,
              bboxes_path: str,
              bboxes_format: str = "csv",  # "csv" o "yolo"
              base_mag_override: Optional[float] = 40.0,
              magnifications: List[int] = [20, 40],
              patch_sizes: List[int] = [512, 768, 1024, 1536],
              overlaps: List[float] = [0.0, 0.25, 0.5],
              margin_frac: float = 0.10,
              out_csv: str = "probe_results.csv") -> pd.DataFrame:


    slide = openslide.OpenSlide(wsi_path)
    w0, h0 = slide.dimensions
    props = get_wsi_props(slide)
    base_mag = base_mag_override or props["objective_power"]
    mpp_x, mpp_y = props["mpp_x"], props["mpp_y"]


    # BBoxes
    if bboxes_format == "csv":
        bxs = load_bboxes_csv(bboxes_path)
    elif bboxes_format == "yolo":
        bxs = load_bboxes_yolo_txt(bboxes_path, w0, h0)
    else:
        raise ValueError("bboxes_format debe ser 'csv' o 'yolo'")


    rows = []
    for mag in magnifications:
        s = scale_factor_for_mag(mag, base_mag, mpp_x, mpp_y)
        for ps in patch_sizes:
            cov_c = coverage_centered(bxs, ps, s, margin_frac=margin_frac)
            for ov in overlaps:
                cov_g = coverage_grid(bxs, w0, h0, ps, s, overlap=ov)
                rows.append({
                    "magnification": mag,
                    "patch_size": ps,
                    "overlap": ov,
                    "scale_from_level0": s,
                    "coverage_centered_%": round(cov_c, 2),
                    "coverage_grid_%": round(cov_g, 2),
                })


    df = pd.DataFrame(rows).sort_values(
        ["magnification","patch_size","overlap"]
    ).reset_index(drop=True)


    df.to_csv(out_csv, index=False)
    print(f"[OK] Resultados guardados en {out_csv}")
    return df


def plot_results(df: pd.DataFrame,
                 metric: str = "coverage_grid_%",
                 out_dir: str = "plots"):
    """
    Crea una linea por magnificación: cobertura vs patch_size,
    con diferentes overlaps como estilos de línea.
    """
    os.makedirs(out_dir, exist_ok=True)
    mags = sorted(df["magnification"].unique())
    ovs  = sorted(df["overlap"].unique())
    pss  = sorted(df["patch_size"].unique())


    for mag in mags:
        sub = df[df["magnification"]==mag].copy()
        plt.figure(figsize=(7,5))
        for ov in ovs:
            sub2 = sub[sub["overlap"]==ov].sort_values("patch_size")
            plt.plot(sub2["patch_size"], sub2[metric], marker="o", label=f"overlap={int(ov*100)}%")
        plt.title(f"{metric} vs patch_size @ {mag}x")
        plt.xlabel("patch_size (px)")
        plt.ylabel(metric)
        plt.xticks(pss)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(out_dir, f"{metric.replace('%','pct')}_mag{mag}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Plot guardado en {out}")


def suggest_best(df: pd.DataFrame) -> pd.Series:
    # Regla simple: máxima cobertura_grid, luego mayor cobertura_centered, y preferir patch size menor
    best = df.sort_values(
        by=["coverage_grid_%","coverage_centered_%","patch_size","magnification"],
        ascending=[False, False, True, True]
    ).iloc[0]
    return best


# ============================
# Extracción de parches (demo)
# ============================


def extract_sample_patches_openslide(wsi_path: str,
                                     centers_level0: List[Tuple[int,int]],
                                     out_dir: str,
                                     desired_mag: int,
                                     patch_size: int,
                                     base_mag_override: Optional[float] = 40.0):
    os.makedirs(out_dir, exist_ok=True)
    slide = openslide.OpenSlide(wsi_path)
    w0, h0 = slide.dimensions
    props = get_wsi_props(slide)
    base_mag = base_mag_override or props["objective_power"]
    s = scale_factor_for_mag(desired_mag, base_mag, props["mpp_x"], props["mpp_y"])


    P0 = int(round(patch_size * s))
    lvl = nearest_level_for_scale(slide, s)
    lvl_down = slide.level_downsamples[lvl]
    P_lvl = int(round(P0 / lvl_down))


    for i, (cx, cy) in enumerate(centers_level0):
        x1 = int(round(cx - P0/2)); y1 = int(round(cy - P0/2))
        x1 = max(0, min(x1, w0 - P0)); y1 = max(0, min(y1, h0 - P0))
        img = slide.read_region((x1, y1), lvl, (P_lvl, P_lvl)).convert("RGB")
        img = img.resize((patch_size, patch_size), resample=Image.BILINEAR)
        img.save(os.path.join(out_dir, f"patch_{i:04d}_mag{desired_mag}_ps{patch_size}.png"))


def extract_sample_patches_tiatoolbox(wsi_path: str,
                                      out_dir: str,
                                      desired_mag: Optional[int] = None,
                                      desired_mpp: Optional[float] = None,
                                      patch_size: int = 1024,
                                      stride: Optional[int] = None,
                                      max_patches: int = 100):
    if not TIATOOLBOX_OK:
        print("TIAToolbox no disponible.")
        return
    os.makedirs(out_dir, exist_ok=True)
    reader = WSIReader.open(wsi_path)
    if desired_mpp is not None:
        res, units = desired_mpp, "mpp"
    elif desired_mag is not None:
        res, units = desired_mag, "power"
    else:
        raise ValueError("Provee desired_mag o desired_mpp.")
    stride = stride or patch_size
    extractor = PatchExtractor(image_reader=reader,
                               resolution=res,
                               units=units,
                               patch_size=patch_size,
                               stride=stride,
                               within_bound=True,
                               pad_mode="constant")
    for i, patch in enumerate(extractor):
        Image.fromarray(patch["image"]).save(os.path.join(out_dir, f"tiatoolbox_{i:05d}.png"))
        if i+1 >= max_patches: break


# ===========
# EJECUCIÓN
# ===========


if __name__ == "__main__":
    # === EDITA AQUÍ ===
    WSI_TIFF     = r"G:\BIOPSIAS RENALES 2025_NL_tiff\BR-090-PAS-25-CONV.tiff"
    BBOXES       = r"G:\yolo_coordenates\BR-090-PAS-25-CONV.txt"   # o "ruta/a/labels.txt" (YOLO)
    BBOX_FORMAT  = "yolo"                      # "csv" o "yolo"


    df = run_probe(
        wsi_path=WSI_TIFF,
        bboxes_path=BBOXES,
        bboxes_format=BBOX_FORMAT,
        base_mag_override=40.0,               # tu escáner base es 40× (AT2)
        magnifications=[20, 40],
        patch_sizes=[512, 768, 1024, 1536],
        overlaps=[0.0, 0.25, 0.5],
        margin_frac=0.10,
        out_csv="probe_results.csv"
    )
    print(df.head())


    # Graficar cobertura (rejilla) y (centrado)
    plot_results(df, metric="coverage_grid_%", out_dir="plots")
    plot_results(df, metric="coverage_centered_%", out_dir="plots")


    # Sugerencia automática
    best = suggest_best(df)
    print("\n=== Sugerencia (regla simple) ===")
    print(best)


    # (Opcional) extraer 20 parches centrados para inspección visual usando la mejor combinación
    # Cargamos bboxes (otra vez) y tomamos centros
    if BBOX_FORMAT == "csv":
        bxs = load_bboxes_csv(BBOXES)
    else:
        slide = openslide.OpenSlide(WSI_TIFF)
        w0,h0 = slide.dimensions
        bxs = load_bboxes_yolo_txt(BBOXES, w0, h0)
    centers = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2) in bxs[:20]]


    extract_sample_patches_openslide(
        wsi_path=WSI_TIFF,
        centers_level0=centers,
        out_dir="patch_samples_openslide",
        desired_mag=int(best["magnification"]),
        patch_size=int(best["patch_size"]),
        base_mag_override=40.0
    )


    # (Opcional) TIAToolbox por magnificación aparente:
    if TIATOOLBOX_OK:
        stride = int(int(best["patch_size"]) * (1.0 - float(best["overlap"])))
        extract_sample_patches_tiatoolbox(
            wsi_path=WSI_TIFF,
            out_dir="patch_samples_tiatoolbox",
            desired_mag=int(best["magnification"]),  # o desired_mpp=0.5 para 20× típico
            patch_size=int(best["patch_size"]),
            stride=stride,
            max_patches=50
        )
