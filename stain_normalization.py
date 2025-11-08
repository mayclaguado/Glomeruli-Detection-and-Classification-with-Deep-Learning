import os
import cv2
import numpy as np
from glob import glob
from shutil import copy2

# ====================== CONFIGURACIÓN ======================
IMAGES_DIR = r"path/to/images"          # tiles
LABELS_DIR = r"path/to/labels"          # .txt (YOLO format)
TARGET_IMG = r"path/to/reference" 
OUT_IMAGES_DIR = r"desired/output_path/images"  # output
OUT_LABELS_DIR = r"desired/output_path/labels"   # output

os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# ====================== UTILIDADES =========================
def bgr_to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

def lab_to_bgr(img_lab_float):
    # asegura rangos válidos antes de convertir
    img_lab = np.clip(img_lab_float, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

def compute_lab_stats(img_bgr):
    lab = bgr_to_lab(img_bgr)
    L, a, b = cv2.split(lab)
    stats = {
        "L_mean": float(np.mean(L)), "L_std": float(np.std(L) + 1e-6),
        "a_mean": float(np.mean(a)), "a_std": float(np.std(a) + 1e-6),
        "b_mean": float(np.mean(b)), "b_std": float(np.std(b) + 1e-6),
    }
    return stats

def reinhard_normalize(src_bgr, target_stats):
    # convierte a Lab y normaliza canal a canal
    lab = bgr_to_lab(src_bgr)
    L, a, b = cv2.split(lab)

    def norm_channel(ch, mu_src=None, std_src=None, mu_t=0., std_t=1.):
        if mu_src is None: mu_src = np.mean(ch)
        if std_src is None: std_src = np.std(ch) + 1e-6
        ch_norm = (ch - mu_src) / std_src
        ch_t = ch_norm * std_t + mu_t
        return ch_t

    L_t = norm_channel(L, mu_t=target_stats["L_mean"], std_t=target_stats["L_std"])
    a_t = norm_channel(a, mu_t=target_stats["a_mean"], std_t=target_stats["a_std"])
    b_t = norm_channel(b, mu_t=target_stats["b_mean"], std_t=target_stats["b_std"])

    lab_t = cv2.merge([L_t, a_t, b_t])
    out_bgr = lab_to_bgr(lab_t)
    return out_bgr

def draw_yolo_boxes(img_bgr, yolo_txt_path, color=(0,255,0), thickness=2, classes=None):
    # verificar que las cajas son iguales en tamaño tras normalizar
    h, w = img_bgr.shape[:2]
    if not os.path.exists(yolo_txt_path):
        return img_bgr
    overlay = img_bgr.copy()
    with open(yolo_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: 
                continue
            cls, xc, yc, bw, bh = parts[:5]
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            # pasar de normalizado a pixeles
            x = int((xc - bw/2) * w)
            y = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            cv2.rectangle(overlay, (x,y), (x2,y2), color, thickness)
            if classes is not None:
                cls_id = int(cls)
                if 0 <= cls_id < len(classes):
                    cv2.putText(overlay, classes[cls_id], (x, max(0,y-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay

# ====================== ESTADÍSTICA OBJETIVO ======================
if not os.path.exists(TARGET_IMG):
    raise FileNotFoundError(f"No se encontró la imagen objetivo: {TARGET_IMG}")

target_bgr = cv2.imread(TARGET_IMG, cv2.IMREAD_COLOR)
if target_bgr is None:
    raise RuntimeError("Error leyendo la imagen objetivo. Verifica ruta y formato.")

target_stats = compute_lab_stats(target_bgr)

# ====================== PROCESAMIENTO  ============================
pngs = sorted(glob(os.path.join(IMAGES_DIR, "*.png")))
if not pngs:
    raise RuntimeError(f"No hay .png en {IMAGES_DIR}")

for i, img_path in enumerate(pngs, 1):
    fname = os.path.basename(img_path)
    base, _ = os.path.splitext(fname)

    # leer imagen de referencia
    src_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if src_bgr is None:
        print(f"[WARN] No pude leer {img_path}, salto.")
        continue

    h0, w0 = src_bgr.shape[:2]

    # normalizar color con Reinhard
    out_bgr = reinhard_normalize(src_bgr, target_stats)

    # validar que dimensiones no cambiaron
    h1, w1 = out_bgr.shape[:2]
    if (h0, w0) != (h1, w1):
        raise RuntimeError("Las dimensiones cambiaron, algo salió mal.")

    # guardar imagen
    out_path = os.path.join(OUT_IMAGES_DIR, fname)
    ok = cv2.imwrite(out_path, out_bgr)
    if not ok:
        raise RuntimeError(f"No pude escribir {out_path}")

    # copiar label correspondiente
    yolo_src = os.path.join(LABELS_DIR, base + ".txt")
    yolo_dst = os.path.join(OUT_LABELS_DIR, base + ".txt")
    if os.path.exists(yolo_src):
        copy2(yolo_src, yolo_dst)

    if i % 25 == 0 or i == len(pngs):
        print(f"Procesadas {i}/{len(pngs)}")

print("Listo: imágenes normalizadas en", OUT_IMAGES_DIR)
print("Labels copiados en", OUT_LABELS_DIR)

# ====================== VERIFICACIÓN DE CAJAS ====================
# dibujar cajas de una imagen de prueba antes y después y exportarlas. Para inspección visual rápida.

CHECK_SAMPLE = None  
if CHECK_SAMPLE is not None:
    src_img_path = os.path.join(IMAGES_DIR, CHECK_SAMPLE + ".png")
    dst_img_path = os.path.join(OUT_IMAGES_DIR, CHECK_SAMPLE + ".png")
    src_lbl_path = os.path.join(LABELS_DIR, CHECK_SAMPLE + ".txt")
    dst_lbl_path = os.path.join(OUT_LABELS_DIR, CHECK_SAMPLE + ".txt")

    src_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
    dst_img = cv2.imread(dst_img_path, cv2.IMREAD_COLOR)
    src_overlay = draw_yolo_boxes(src_img, src_lbl_path, color=(0,255,0))
    dst_overlay = draw_yolo_boxes(dst_img, dst_lbl_path, color=(0,0,255))

    cv2.imwrite(os.path.join(OUT_IMAGES_DIR, CHECK_SAMPLE + "_src_boxes.png"), src_overlay)
    cv2.imwrite(os.path.join(OUT_IMAGES_DIR, CHECK_SAMPLE + "_dst_boxes.png"), dst_overlay)
    print("Guardadas superposiciones de verificación para", CHECK_SAMPLE)