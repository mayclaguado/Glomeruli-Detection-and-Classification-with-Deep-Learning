import json, os, unicodedata
from collections import Counter
from PIL import Image
try:
    import openslide
    HAS_OS = True
except Exception:
    HAS_OS = False

# ========= CONFIG =========
geojson_path = r"geojson new images/BR-076-PAS-25-CONV.geojson"
image_path   = r"BR-076-PAS-25-CONV.tiff"

# Carpeta fija donde se guardarán SIEMPRE los .txt
OUTPUT_DIR = r"C:\Users\mayca\OneDrive\Escritorio\coordenadas new images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

show_examples_unknown = True  # imprime ejemplos de etiquetas no reconocidas

# ======= NORMALIZACIÓN =======
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def norm_text(s: str) -> str:
    s = strip_accents(s).lower().strip()
    # quita comillas/puntuación comunes y colapsa espacios
    s = s.strip(" '\"`´“”‘’.,:;!-_()[]{}")
    for ch in "'\"`´“”‘’.,:;!()[]{}":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

# Mapeo por raíces (masculino/femenino/plural)
def map_label_to_id(lbl: str | None):
    if not isinstance(lbl, str):
        return None
    n = norm_text(lbl)
    # orden importa: primero "no proliferativ" para no confundir con "proliferativ"
    if "no proliferativ" in n:
        return 0  # no proliferativo/a/os/as
    if "proliferativ" in n:
        return 1  # proliferativo/a/os/as
    if "esclerosad" in n:
        return 2  # esclerosado/a/os/as
    if "exclu" in n:   # exclude / excluir / excluido/a
        return 3
    return None

ID_TO_NAME = {
    0: "no proliferativo",
    1: "proliferativo",
    2: "esclerosado",
    3: "exclude",
}

# ======= DIMENSIONES W,H =======
def get_wh(path):
    if HAS_OS:
        try:
            s = openslide.OpenSlide(path)
            return s.dimensions
        except Exception:
            pass
    return Image.open(path).size

W, H = get_wh(image_path)

# ======== UTILIDADES GEO ========
def poly_to_yolo(coords, W, H):
    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    xc = ((x0 + x1)/2) / W
    yc = ((y0 + y1)/2) / H
    w  = (x1 - x0) / W
    h  = (y1 - y0) / H
    return xc, yc, w, h

# ====== EXTRACCIÓN DE ETIQUETA ======
COMMON_PATHS = [
    ["classification","name"],      # QuPath común
    ["classifications","0","name"], # variante
    ["name"], ["label"], ["class"], ["title"], ["category"], ["tipo"], ["grupo"],
]

def get_from_path(d, tokens):
    cur = d
    for t in tokens:
        if isinstance(cur, dict) and t in cur:
            cur = cur[t]
        elif isinstance(cur, list):
            try:
                cur = cur[int(t)]
            except Exception:
                return None
        else:
            return None
    return cur

def iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)

def extract_label(props: dict):
    # 1) rutas comunes
    for path in COMMON_PATHS:
        v = get_from_path(props, path)
        if isinstance(v, str) and v.strip():
            return v
    # 2) fallback: cualquier string plausible
    for s in iter_strings(props):
        if map_label_to_id(s) is not None:
            return s
    return None

# ========== PROCESO ÚNICO: AUDITA + CONVIERTE ==========
with open(geojson_path, "r", encoding="utf-8") as f:
    gj = json.load(f)

base = os.path.splitext(os.path.basename(image_path))[0]
out_txt = os.path.join(OUTPUT_DIR, base + ".txt")

# Auditoría
found_norm_labels = []
for feat in gj.get("features", []):
    props = feat.get("properties", {}) or {}
    lbl = extract_label(props)
    found_norm_labels.append("(sin etiqueta)" if lbl is None else norm_text(lbl))

print("=== Resumen de etiquetas encontradas (normalizadas) ===")
for k, v in Counter(found_norm_labels).most_common():
    print(f"{k}: {v}")
print("=======================================================\n")

# Conversión
lines = []
total = written = unknown = 0
unknown_examples = []

for feat in gj.get("features", []):
    total += 1
    props = feat.get("properties", {}) or {}
    geom  = feat.get("geometry", {}) or {}

    raw  = extract_label(props)
    cid  = map_label_to_id(raw)
    if cid is None:
        unknown += 1
        if show_examples_unknown and len(unknown_examples) < 8:
            unknown_examples.append(repr(raw))
        continue

    gtype = geom.get("type")
    coords = geom.get("coordinates")

    if gtype == "Polygon" and coords:
        xc, yc, w, h = poly_to_yolo(coords[0], W, H)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        written += 1
    elif gtype == "MultiPolygon" and coords:
        for poly in coords:
            xc, yc, w, h = poly_to_yolo(poly[0], W, H)
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            written += 1

print("Primeras líneas que se guardarán:")
for s in lines[:10]:
    print(s)

# Guardar en carpeta fija
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n✅ Guardado en carpeta fija:")
print(os.path.abspath(out_txt))
print(f"Total features: {total} | Escritas: {written} | Sin clase reconocida: {unknown}")

print("\nIDs → nombres (para data.yaml):")
for i, n in ID_TO_NAME.items():
    print(f"{i}: {n}")
