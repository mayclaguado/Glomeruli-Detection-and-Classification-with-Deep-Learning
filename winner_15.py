# -*- coding: utf-8 -*-
"""
WORKED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WSI -> Tiles (streaming) -> Dataset YOLO (PNG) con filtro de tejido adaptativo

Nueva política:
- single_winner_per_glom:
    * Elige 1 (y sólo 1) tile ganador por glomérulo (>= beta, o el mejor si >= gamma_partial).
    * Se conservan únicamente tiles que tengan al menos un ganador.
    * En tiles ganadores se añaden también parciales de otros glomérulos si visibilidad >= alpha_partial
      (excepción de co-ocurrencia que pediste).
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import openslide
from PIL import Image, ImageFilter
import random

# ---------------------------
# Utilidades geométricas
# ---------------------------

def _clip_box_to_tile(x1, y1, x2, y2, tx, ty, tw, th):
    orig_w = max(0.0, x2 - x1)
    orig_h = max(0.0, y2 - y1)
    if orig_w <= 0 or orig_h <= 0:
        return None
    ix1 = max(x1, tx); iy1 = max(y1, ty)
    ix2 = min(x2, tx + tw); iy2 = min(y2, ty + th)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih
    if inter_area <= 0:
        return None
    visible_frac = inter_area / (orig_w * orig_h)
    return ix1, iy1, ix2, iy2, visible_frac

# ---------------------------
# Anotaciones
# ---------------------------

def load_yolo_annotations_normalized(wsi_path: str) -> List[List[float]]:
    annot_path = Path(wsi_path).with_suffix('.txt')
    if not annot_path.exists():
        raise FileNotFoundError(f"No se encontró archivo YOLO: {annot_path}")
    with openslide.OpenSlide(wsi_path) as slide:
        W, H = slide.dimensions

    anns = []
    with open(annot_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, cx, cy, w, h = map(float, parts)
            cx_abs = cx * W; cy_abs = cy * H
            w_abs = w * W;   h_abs = h * H
            x1 = cx_abs - w_abs/2; y1 = cy_abs - h_abs/2
            x2 = cx_abs + w_abs/2; y2 = cy_abs + h_abs/2
            anns.append([x1, y1, x2, y2, int(class_id)])
    return anns

def scale_annotations_to_downsample(annotations_abs: List[List[float]], scale_used: float):
    return [[x1/scale_used, y1/scale_used, x2/scale_used, y2/scale_used, cid]
            for x1,y1,x2,y2,cid in annotations_abs]

# ---------------------------
# Grilla (offsets; sin imágenes)
# ---------------------------

def build_tile_grid_offsets(wsi_path, target_downsample=2.0, tile_size=(1536,1536), overlap=0.6):
    tw, th = tile_size
    with openslide.OpenSlide(wsi_path) as slide:
        W0, H0 = slide.dimensions
    Wt = int(W0 / target_downsample); Ht = int(H0 / target_downsample)

    step_x = max(1, int(tw * (1 - overlap)))
    step_y = max(1, int(th * (1 - overlap)))
    xs = list(range(0, max(Wt - tw + 1, 1), step_x))
    ys = list(range(0, max(Ht - th + 1, 1), step_y))
    if xs[-1] != Wt - tw: xs.append(Wt - tw)
    if ys[-1] != Ht - th: ys.append(Ht - th)
    return xs, ys, tw, th

# ---------------------------
# Políticas (con offsets)
# ---------------------------

def _t_index(ix, iy, nx): return iy*nx + ix

def choose_winner_tiles_unique_filtered(
    annotations_down, xs, ys, tw, th,
    beta=0.6, alpha=0.2, gamma_partial=0.35,
    label_partials_in_kept_tiles=True, edge_margin=0
):
    nx = len(xs)
    per_obj_hits = []
    for (x1,y1,x2,y2,cid) in annotations_down:
        hits = []
        best = None
        for iy, ty in enumerate(ys):
            for ix, tx in enumerate(xs):
                clip = _clip_box_to_tile(x1,y1,x2,y2, tx,ty, tw,th)
                if not clip: continue
                ix1,iy1,ix2,iy2,vis = clip
                t_idx = _t_index(ix,iy,nx)
                hits.append((t_idx, vis, (ix1,iy1,ix2,iy2), (tx,ty)))
                if (best is None) or (vis > best[1]):
                    best = (t_idx, vis, (ix1,iy1,ix2,iy2), (tx,ty))
        per_obj_hits.append((x1,y1,x2,y2,cid,hits,best))

    dets_by_tile: Dict[int, List[List[float]]] = {}
    tiles_to_exclude: Set[int] = set()

    for (x1,y1,x2,y2,cid,hits,best) in per_obj_hits:
        if not hits or best is None: continue
        winners = []
        for (t_idx, vis, (ix1,iy1,ix2,iy2), (tx,ty)) in hits:
            if vis < beta: continue
            if edge_margin > 0:
                x1_t, y1_t = ix1 - tx, iy1 - ty
                x2_t, y2_t = ix2 - tx, iy2 - ty
                if (x1_t < edge_margin or y1_t < edge_margin or
                    (tw - x2_t) < edge_margin or (th - y2_t) < edge_margin):
                    continue
            winners.append(t_idx)

        accepted: Set[int] = set()
        if winners:
            for t_idx in winners:
                dets_by_tile.setdefault(t_idx, []).append([x1,y1,x2,y2,cid])
            accepted = set(winners)
        else:
            t_best, vis_best, _, _ = best
            if vis_best >= gamma_partial:
                dets_by_tile.setdefault(t_best, []).append([x1,y1,x2,y2,cid])
                accepted = {t_best}

        if accepted:
            for (t_idx, vis, _, _) in hits:
                if (t_idx not in accepted) and (vis >= alpha):
                    tiles_to_exclude.add(t_idx)

    if label_partials_in_kept_tiles:
        kept_tiles = set(dets_by_tile.keys())
        for (x1,y1,x2,y2,cid,hits,best) in per_obj_hits:
            if not hits: continue
            for (t_idx, vis, _, _) in hits:
                if (t_idx in kept_tiles) and (vis >= alpha):
                    already = dets_by_tile.setdefault(t_idx, [])
                    if all(not (abs(ax1-x1)<1e-6 and abs(ay1-y1)<1e-6 and abs(ax2-x2)<1e-6 and abs(ay2-y2)<1e-6 and ac==cid)
                           for ax1,ay1,ax2,ay2,ac in already):
                        already.append([x1,y1,x2,y2,cid])

    all_tiles = set(_t_index(ix,iy,nx) for iy in range(len(ys)) for ix in range(len(xs)))
    tiles_with_dets = set(dets_by_tile.keys())
    tiles_to_keep = (all_tiles - tiles_to_exclude) | tiles_with_dets
    return tiles_to_keep, dets_by_tile

def choose_single_winner_per_glom(
    annotations_down, xs, ys, tw, th,
    beta=0.70,               # pide "casi completo"
    gamma_partial=0.35,      # si nadie llega a beta, acepta mejor si >= gamma_partial
    alpha_partial=0.35,      # parciales a anotar solo en tiles que ya son ganadoras de algún glom
    edge_margin=0
):
    """
    1) Para cada glom, elige un ÚNICO tile ganador:
       - Si hay tiles con vis >= beta, elige la de mayor vis entre ellas.
       - Si no, elige la de mayor vis si >= gamma_partial.
    2) Se conservan únicamente los tiles que tengan >=1 ganador.
    3) En esos tiles ganadores, también se anotan los OTROS glomérulos que caen parcial
       (vis >= alpha_partial), cumpliendo tu excepción de co-ocurrencia.
    """
    nx = len(xs)

    # Recolecta impactos por objeto
    per_obj_hits = []
    for (x1,y1,x2,y2,cid) in annotations_down:
        hits = []  # (t_idx, vis, (ix1,iy1,ix2,iy2), (tx,ty))
        for iy, ty in enumerate(ys):
            for ix, tx in enumerate(xs):
                clip = _clip_box_to_tile(x1,y1,x2,y2, tx,ty, tw,th)
                if not clip: continue
                ix1,iy1,ix2,iy2,vis = clip
                t_idx = _t_index(ix,iy,nx)
                hits.append((t_idx, vis, (ix1,iy1,ix2,iy2), (tx,ty)))
        per_obj_hits.append((x1,y1,x2,y2,cid,hits))

    # Determina ganador único por objeto
    winner_by_obj: Dict[int, int] = {}  # key = index obj, value = t_idx ganador
    for i,(x1,y1,x2,y2,cid,hits) in enumerate(per_obj_hits):
        if not hits:
            continue
        # candidatos >= beta
        candidates = [h for h in hits if h[1] >= beta]
        if candidates:
            # el de mayor visibilidad
            t_idx = max(candidates, key=lambda h: h[1])[0]
            winner_by_obj[i] = t_idx
        else:
            # mejor si >= gamma_partial
            t_best, vis_best, _, _ = max(hits, key=lambda h: h[1])
            if vis_best >= gamma_partial:
                winner_by_obj[i] = t_best
            # si no supera gamma_partial, no tendrá tile en el dataset

    # Construye labels por tile: sólo ganadores + parciales en tiles ganadores
    dets_by_tile: Dict[int, List[List[float]]] = {}
    tiles_with_any_winner: Set[int] = set(winner_by_obj.values())

    # 2.a) Añadir ganadores
    for i,(x1,y1,x2,y2,cid,hits) in enumerate(per_obj_hits):
        if i not in winner_by_obj: 
            continue
        t_w = winner_by_obj[i]
        dets_by_tile.setdefault(t_w, []).append([x1,y1,x2,y2,cid])

    # 2.b) Añadir parciales (excepción) SOLO en tiles que ya son ganadores de alguien
    for (x1,y1,x2,y2,cid,hits) in per_obj_hits:
        for (t_idx, vis, _, _) in hits:
            if (t_idx in tiles_with_any_winner) and (vis >= alpha_partial):
                already = dets_by_tile.setdefault(t_idx, [])
                # evita duplicar si ya está como ganador
                if all(not (abs(ax1-x1)<1e-6 and abs(ay1-y1)<1e-6 and abs(ax2-x2)<1e-6 and abs(ay2-y2)<1e-6 and ac==cid)
                       for ax1,ay1,ax2,ay2,ac in already):
                    already.append([x1,y1,x2,y2,cid])

    tiles_to_keep = set(tiles_with_any_winner)  # sólo tiles con ganadores
    return tiles_to_keep, dets_by_tile

def build_dets_all_intersections_offsets(annotations_down, xs, ys, tw, th):
    nx = len(xs)
    dets_by_tile: Dict[int, List[List[float]]] = {}
    for iy, ty in enumerate(ys):
        for ix, tx in enumerate(xs):
            t_idx = _t_index(ix,iy,nx)
            for x1,y1,x2,y2,cid in annotations_down:
                if _clip_box_to_tile(x1,y1,x2,y2, tx,ty, tw,th):
                    dets_by_tile.setdefault(t_idx, []).append([x1,y1,x2,y2,cid])
    tiles_to_keep = set(_t_index(ix,iy,nx) for iy in range(len(ys)) for ix in range(len(xs)))
    return tiles_to_keep, dets_by_tile

def compute_tiles_with_any_intersection(annotations_down, xs, ys, tw, th) -> Set[int]:
    nx = len(xs)
    touched: Set[int] = set()
    for iy, ty in enumerate(ys):
        for ix, tx in enumerate(xs):
            t_idx = _t_index(ix, iy, nx)
            for x1,y1,x2,y2,_ in annotations_down:
                if _clip_box_to_tile(x1,y1,x2,y2, tx,ty, tw,th):
                    touched.add(t_idx)
                    break
    return touched

# ---------------------------
# Tissue mask (thumbnail HSV + Otsu en S + distancia al beige)
# ---------------------------

def _otsu_threshold(values_0_255: np.ndarray) -> int:
    hist, _ = np.histogram(values_0_255, bins=256, range=(0,256))
    total = values_0_255.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b, w_b, max_var, thresh = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    return int(thresh)

def build_tissue_mask_quick(wsi_path, short_side=2048, sat_floor=20,
                            val_min=25, val_max=245, beige_dist=18,
                            preserve_small_islands=True):
    with openslide.OpenSlide(wsi_path) as slide:
        W0, H0 = slide.dimensions
        if W0 <= H0:
            tw = short_side
            th = int(H0 * (tw / W0))
        else:
            th = short_side
            tw = int(W0 * (th / H0))
        thumb = slide.get_thumbnail((tw, th)).convert("RGB")

    arr = np.array(thumb)

    corners = np.concatenate([
        arr[0:64, 0:64].reshape(-1,3),
        arr[0:64, -64:].reshape(-1,3),
        arr[-64:, 0:64].reshape(-1,3),
        arr[-64:, -64:].reshape(-1,3)
    ], axis=0)
    bg = corners.mean(axis=0)

    hsv = np.array(thumb.convert("HSV"))
    S = hsv[...,1]; V = hsv[...,2]

    thr_otsu = _otsu_threshold(S)
    sat_thr = max(thr_otsu, sat_floor)
    mask_sat = (S >= sat_thr)
    mask_val = (V >= val_min) & (V <= val_max)

    diff = arr.astype(np.float32) - bg.reshape(1,1,3)
    dist = np.linalg.norm(diff, axis=2)
    mask_not_beige = dist >= beige_dist

    mask = (mask_sat & mask_val & mask_not_beige).astype(np.uint8)

    mimg = Image.fromarray((mask*255).astype("uint8"))
    if preserve_small_islands:
        mimg = mimg.filter(ImageFilter.MaxFilter(3))
        mimg = mimg.filter(ImageFilter.MinFilter(3))
    else:
        mimg = mimg.filter(ImageFilter.MinFilter(3))
        mimg = mimg.filter(ImageFilter.MaxFilter(5))
    mask = (np.array(mimg) > 0).astype(np.uint8)

    return mask, tw, th, W0, H0

def dilate_mask(mask: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    img = Image.fromarray((mask*255).astype("uint8"))
    for _ in range(iters):
        img = img.filter(ImageFilter.MaxFilter(k))
    return (np.array(img) > 0).astype(np.uint8)

def tile_coverage(mask, tx, ty, tw, th, sx, sy):
    x1 = int(round(tx * sx)); y1 = int(round(ty * sy))
    x2 = int(round((tx + tw) * sx)); y2 = int(round((ty + th) * sy))
    x1 = max(0, min(mask.shape[1]-1, x1)); x2 = max(0, min(mask.shape[1], x2))
    y1 = max(0, min(mask.shape[0]-1, y1)); y2 = max(0, min(mask.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    return float(patch.sum()) / float(patch.size)

# ---------------------------
# Export streaming (PNG)
# ---------------------------

def export_streaming_png(
    wsi_path, target_downsample,
    xs, ys, tw, th,
    tiles_to_keep, dets_by_tile,
    out_root="dataset",
    filename_prefix="tile", start_index=0,
    keep_empty_labels=True
):
    out_root = Path(out_root)
    img_root = out_root / "images"; img_root.mkdir(parents=True, exist_ok=True)
    lbl_root = out_root / "labels"; lbl_root.mkdir(parents=True, exist_ok=True)

    def name_for(idx): return f"{filename_prefix}_{idx + start_index:06d}"

    with openslide.OpenSlide(wsi_path) as slide:
        level = slide.get_best_level_for_downsample(target_downsample)
        dL = float(slide.level_downsamples[level])
        r = target_downsample / dL

        nx = len(xs)
        t_idx = -1
        for iy, ty in enumerate(ys):
            for ix, tx in enumerate(xs):
                t_idx += 1
                if t_idx not in tiles_to_keep:
                    continue

                base_x = int(tx * target_downsample)
                base_y = int(ty * target_downsample)
                rw = max(1, int(round(tw * r)))
                rh = max(1, int(round(th * r)))
                region = slide.read_region((base_x, base_y), level, (rw, rh)).convert("RGB")
                if region.size != (tw, th):
                    region = region.resize((tw, th), Image.BILINEAR)

                name = name_for(t_idx)
                img_path = img_root / f"{name}.png"
                lbl_path = lbl_root / f"{name}.txt"
                region.save(img_path, "PNG", optimize=True, compress_level=6)

                lines = []
                for (x1,y1,x2,y2,cid) in dets_by_tile.get(t_idx, []):
                    ix1=max(x1,tx); iy1=max(y1,ty); ix2=min(x2,tx+tw); iy2=min(y2,ty+th)
                    if ix2<=ix1 or iy2<=iy1: 
                        continue
                    cx=(ix1+ix2)/2 - tx; cy=(iy1+iy2)/2 - ty
                    w=(ix2-ix1);         h=(iy2-iy1)
                    cx/=tw; cy/=th; w/=tw; h/=th
                    cx=min(max(cx,0.0),1.0); cy=min(max(cy,0.0),1.0)
                    w =min(max(w, 1e-6),1.0); h =min(max(h, 1e-6),1.0)
                    lines.append(f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if lines or keep_empty_labels:
                    lbl_path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------
# Orquestador
# ---------------------------

def pipeline_wsi_to_yolo_dataset_streaming(
    wsi_path: str,
    out_root: str,
    tile_size=(1536, 1536),
    overlap=0.6,
    target_downsample=2.0,
    policy="single_winner_per_glom",
    beta=0.7,                 # umbral de "casi completo"
    alpha=0.35,               # alpha_partial para co-ocurrencias en tiles ganadoras
    filename_prefix="tile",
    start_index=0,
    keep_empty_labels=True,   # <-- deja True para .txt vacíos de negativos
    # Tissue filter
    tissue_filter=True,
    tau=0.12,                 # cobertura mínima de tejido (0..1)
    short_side=2048,
    sat_thresh=22,
    val_min=25,
    val_max=245,
    beige_dist=18,
    preserve_small_islands=True,
    dilate_for_coverage=True,
    dilate_k=3,
    dilate_iters=1
):
    # 1) anotaciones (original -> target)
    annotations_abs  = load_yolo_annotations_normalized(wsi_path)
    annotations_down = scale_annotations_to_downsample(annotations_abs, target_downsample)

    # 2) grilla de offsets
    xs, ys, tw, th = build_tile_grid_offsets(
        wsi_path, target_downsample=target_downsample, tile_size=tile_size, overlap=overlap
    )
    nx = len(xs)
    all_tiles = set(iy*nx + ix for iy in range(len(ys)) for ix in range(len(xs)))

    # 3) política (1 ganador por glom + co-ocurrencias dentro de tiles ganadoras)
    if policy == "single_winner_per_glom":
        tiles_to_keep, dets_by_tile = choose_single_winner_per_glom(
            annotations_down, xs, ys, tw, th,
            beta=beta, gamma_partial=0.35, alpha_partial=alpha, edge_margin=0
        )
    elif policy == "unique_filtered":
        tiles_to_keep, dets_by_tile = choose_winner_tiles_unique_filtered(
            annotations_down, xs, ys, tw, th,
            beta=beta, alpha=alpha, gamma_partial=0.35,
            label_partials_in_kept_tiles=True, edge_margin=0
        )
    elif policy == "all_intersections":
        tiles_to_keep, dets_by_tile = build_dets_all_intersections_offsets(
            annotations_down, xs, ys, tw, th
        )
    else:
        raise ValueError("policy debe ser 'single_winner_per_glom', 'unique_filtered' o 'all_intersections'")

    # 3.b) tiles que tocan algún glom y negativos puros (no tocan ninguno)
    tiles_touching_any = compute_tiles_with_any_intersection(
        annotations_down, xs, ys, tw, th
    )
    pure_negatives = all_tiles - tiles_touching_any

    # 4) Máscara de tejido SIEMPRE (querías mantenerla)
    #    - Tiles con labels: se conservan siempre (no filtrarlas por tejido).
    #    - Negativos: se conservan SOLO si cobertura de tejido >= tau.
    cov_by_tile = {}  # *** NUEVO: cobertura por tile (Solo NEGATIVOS candidatos) ***
    if tissue_filter:
        mask, tw_thumb, th_thumb, W0, H0 = build_tissue_mask_quick(
            wsi_path,
            short_side=short_side,
            sat_floor=sat_thresh,
            val_min=val_min, val_max=val_max,
            beige_dist=beige_dist,
            preserve_small_islands=preserve_small_islands
        )
        Wt = W0 / target_downsample; Ht = H0 / target_downsample
        sx = tw_thumb / Wt; sy = th_thumb / Ht
        mask_for_cov = dilate_mask(mask, k=dilate_k, iters=dilate_iters) if dilate_for_coverage else mask

        tiles_with_labels = set(dets_by_tile.keys())

        # 4.a) Empieza con todas las tiles con labels (¡no las perdemos!)
        filtered_keep: Set[int] = set(tiles_with_labels)

        # 4.b) Negativos ya propuestos por la política: filtra por tejido
        for t_idx in (tiles_to_keep - tiles_with_labels):
            iy, ix = divmod(t_idx, nx)
            tx, ty = xs[ix], ys[iy]
            cov = tile_coverage(mask_for_cov, tx, ty, tw, th, sx, sy)
            cov_by_tile[t_idx] = cov  # *** NUEVO: guardar cobertura de NEGATIVO ***
            if cov >= tau:
                filtered_keep.add(t_idx)

        # 4.c) Añade TODOS los negativos puros con tejido (para cumplir tu requisito)
        for t_idx in pure_negatives:
            iy, ix = divmod(t_idx, nx)
            tx, ty = xs[ix], ys[iy]
            cov = tile_coverage(mask_for_cov, tx, ty, tw, th, sx, sy)
            cov_by_tile[t_idx] = cov  # *** NUEVO ***
            if cov >= tau:
                filtered_keep.add(t_idx)

        tiles_to_keep = filtered_keep

    # 5) Eliminar "negativos sucios": tiles que tocan glom pero quedaron sin labels
    tiles_with_labels = set(dets_by_tile.keys())
    dirty_negatives = {t for t in tiles_to_keep if (t in tiles_touching_any) and (t not in tiles_with_labels)}
    tiles_to_keep = tiles_to_keep - dirty_negatives

    # *** NUEVO: Muestrear NEGATIVOS = round(0.15 * #POS) con 80% 'hard' estocástico + 20% aleatorio ***
    pos_tiles = tiles_to_keep & tiles_with_labels
    neg_tiles = list(tiles_to_keep - tiles_with_labels)

    n_pos = len(pos_tiles)
    quota_neg = int(round(0.15 * n_pos))

    if quota_neg < len(neg_tiles) and quota_neg > 0:
        # 80% 'hard' (alto tejido) pero aleatorio dentro de un pool superior; 20% aleatorio puro
        k_top = int(round(0.80 * quota_neg))
        k_rand = max(0, quota_neg - k_top)

        # Orden base por cobertura (si no hay cobertura registrada, usa 0.0)
        neg_sorted = sorted(neg_tiles, key=lambda t: cov_by_tile.get(t, 0.0), reverse=True)

        # --- Pool 'hard' estocástico ---
        hard_pool_size = max(k_top, min(len(neg_sorted), int(max(3*k_top, k_top))))
        hard_pool = neg_sorted[:hard_pool_size]

        # Diversidad espacial: evita vecinos inmediatos si es posible
        def _ixiy(t):
            return (t % nx, t // nx)
        def _far_enough(sel_set, cand, min_md=2):
            cx, cy = _ixiy(cand)
            for s in sel_set:
                sx, sy = _ixiy(s)
                if abs(cx - sx) + abs(cy - sy) < min_md:  # Manhattan < 2 => vecino inmediato
                    return False
            return True

        rng_pool = list(hard_pool)
        random.shuffle(rng_pool)
        top_keep = set()
        # 1) llenar evitando vecinos
        for t in rng_pool:
            if len(top_keep) >= k_top:
                break
            if _far_enough(top_keep, t, min_md=2):
                top_keep.add(t)
        # 2) si faltan, completar sin restricción
        if len(top_keep) < k_top:
            remaining = [t for t in rng_pool if t not in top_keep]
            need = k_top - len(top_keep)
            if remaining:
                top_keep.update(random.sample(remaining, k=min(need, len(remaining))))

        # 3) 20% aleatorio puro del resto
        remaining = list(set(neg_tiles) - top_keep)
        random_keep = set()
        if k_rand > 0 and remaining:
            random_keep = set(random.sample(remaining, k=min(k_rand, len(remaining))))

        selected_negatives = top_keep | random_keep
        tiles_to_keep = pos_tiles | selected_negatives
    # Si quota_neg >= len(neg_tiles) o quota_neg==0, no se modifica (ya hay pocos negativos)


    # 6) Exportar PNG + .txt (negativos tendrán .txt vacío si keep_empty_labels=True)
    export_streaming_png(
        wsi_path, target_downsample,
        xs, ys, tw, th,
        tiles_to_keep, dets_by_tile,
        out_root=out_root,
        filename_prefix=filename_prefix,
        start_index=start_index,
        keep_empty_labels=keep_empty_labels
    )


# ======================
# EJEMPLO DE USO
# ======================
if __name__ == "__main__":
    wsi_path = r"C:\Users\mayca\OneDrive\Escritorio\prueba de imagenes en qupath\BR-076-PAS-25-CONV.tiff"
    out_root = r"C:\Users\mayca\OneDrive\Escritorio\INFFFFF"

    pipeline_wsi_to_yolo_dataset_streaming(
        wsi_path=wsi_path,
        out_root=out_root,
        tile_size=(1536, 1536),
        overlap=0.6,
        target_downsample=2.0,
        policy="single_winner_per_glom",   # <--- usa la nueva política
        beta=0.70,
        alpha=0.35,
        filename_prefix=Path(wsi_path).stem,
        start_index=0,
        keep_empty_labels=True,
        tissue_filter=True,
        tau=0.12,
        short_side=2048,
        sat_thresh=22,
        val_min=25,
        val_max=245,
        beige_dist=18,
        preserve_small_islands=True,
        dilate_for_coverage=True,
        dilate_k=3,
        dilate_iters=1
    )

    print("✅ Dataset YOLO (PNG) exportado en:", out_root)
