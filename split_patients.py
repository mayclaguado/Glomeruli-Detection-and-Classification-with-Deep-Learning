import shutil
import random
from pathlib import Path
from collections import defaultdict

# ============ CONFIGURACIÓN ============
ROOT = Path("/kaggle/working/norm_dataset")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"

# TUS PROPORCIONES (Deben sumar 1.0)
TRAIN_RATIO = 0.70  # Aprendizaje
VAL_RATIO   = 0.05  # Monitoreo rápido
TEST_RATIO  = 0.25  # Prueba de fuego final

# Carpeta de salida
OUT_DIR = Path(f"/kaggle/working/split_patient_FINAL_70_5_25")
CLASS_NAMES = ["no_proliferativo", "proliferativo", "esclerosado", "exclude"] 

# ============ 1. AGRUPAR POR PACIENTE ============
print(f"--- INICIANDO PROCESO ---")
print(f"Configuración: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")

supported_exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
images = []
for ext in supported_exts:
    images.extend(list(IMG_DIR.glob(ext)))

# Diccionario para agrupar: { 'ID_PACIENTE': [lista_de_sus_fotos] }
patient_map = defaultdict(list)

for img_path in images:
    # Lógica: Cortar en el primer guion bajo
    # 'BR-94-PAS-24-CONV_001948_augPURE' -> ID: 'BR-94-PAS-24-CONV'
    patient_id = img_path.stem.split('_')[0]
    patient_map[patient_id].append(img_path)

all_patients = list(patient_map.keys())
random.seed(42) # Semilla fija para reproducibilidad
random.shuffle(all_patients)

total_pats = len(all_patients)
print(f"✅ Total Pacientes únicos: {total_pats}")
print(f"✅ Total Imágenes: {len(images)}")

# ============ 2. CALCULAR CORTES ============
# Calculamos cuántos pacientes van a cada grupo
n_train = int(total_pats * TRAIN_RATIO)
n_val   = int(total_pats * VAL_RATIO)
# El resto a test para asegurar que sume el total
n_test  = total_pats - n_train - n_val

# Asignar las listas de pacientes
patients_train = all_patients[:n_train]
patients_val   = all_patients[n_train : n_train + n_val]
patients_test  = all_patients[n_train + n_val:]

# Comprobación de seguridad (Anti-Leakage)
assert len(set(patients_train) & set(patients_val)) == 0
assert len(set(patients_train) & set(patients_test)) == 0
assert len(set(patients_val) & set(patients_test)) == 0

print(f"\n--- DISTRIBUCIÓN DE PACIENTES ---")
print(f"TRAIN: {len(patients_train)} pacientes")
print(f"VAL:    {len(patients_val)} pacientes (Mínimo para monitoreo)")
print(f"TEST:  {len(patients_test)} pacientes (Evaluación final)")

# ============ 3. COPIAR ARCHIVOS FÍSICAMENTE ============
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR) # Limpiar si existe

def copy_subset(patient_list, split_name):
    # Crear subcarpetas images/labels
    img_dest = OUT_DIR / split_name / "images"
    lbl_dest = OUT_DIR / split_name / "labels"
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for pid in patient_list:
        # Copiar todas las fotos de ese paciente (originales + aumentadas)
        for img_p in patient_map[pid]:
            # Copiar imagen
            shutil.copy2(img_p, img_dest / img_p.name)
            
            # Copiar label
            lbl_p = LBL_DIR / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.copy2(lbl_p, lbl_dest / f"{img_p.stem}.txt")
            else:
                # Si es imagen de fondo (sin objetos), crear txt vacío
                (lbl_dest / f"{img_p.stem}.txt").touch()
            count += 1
    return count

print(f"\n--- COPIANDO ARCHIVOS ---")
cnt_train = copy_subset(patients_train, "train")
cnt_val   = copy_subset(patients_val,   "val")
cnt_test  = copy_subset(patients_test,  "test")

print(f"Imágenes en TRAIN: {cnt_train}")
print(f"Imágenes en VAL:   {cnt_val}")
print(f"Imágenes en TEST:  {cnt_test}")

# ============ 4. CREAR DATASET.YAML ============
# Configuramos el YAML para que YOLO entienda los 3 conjuntos
yaml_content = f"""path: {OUT_DIR.as_posix()}
train: train/images
val: val/images
test: test/images
names:
  0: {CLASS_NAMES[0]}
  1: {CLASS_NAMES[1]}
  2: {CLASS_NAMES[2]}
  3: {CLASS_NAMES[3]}
"""

(OUT_DIR / "dataset.yaml").write_text(yaml_content)

print(f"\n✅ ¡Listo! Dataset preparado en: {OUT_DIR}")
print(f"   Archivo de configuración: {OUT_DIR / 'dataset.yaml'}")