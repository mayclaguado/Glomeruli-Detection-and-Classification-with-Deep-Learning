# Glomeruli-Detection-and-Classification-with-Deep-Learning
Turns WSIs into glomeruli findings: downsampling, tissue masking, and overlapping tiling. YOLOv8 detects, we map results to slide coordinates, flag complete vs edge boxes, and deduplicate for a reliable count. All this for a 4-class LN classifier. Finally, it exports color overlays and audit-ready tables.

🩺 Sistema Automatizado de Diagnóstico de Nefritis Lúpica

Un sistema completo de inteligencia artificial para el diagnóstico automatizado de nefritis lúpica en biopsias renales mediante detección y clasificación de glomérulos.

Python 3.8+ PyTorch License: MIT
📋 Descripción del Proyecto

Este sistema implementa un pipeline completo de visión por computadora para el diagnóstico automatizado de nefritis lúpica, una complicación renal del lupus eritematoso sistémico. El sistema:

    Detecta automáticamente glomérulos en imágenes TIFF de alta resolución de biopsias renales
    Clasifica cada glomérulo en una de 3 clases agrupadas de nefritis lúpica
    Genera un diagnóstico final de la biopsia basado en la clasificación predominante

🎯 Características Principales

    ✅ Detección robusta con YOLOv8 optimizado para imágenes histológicas
    ✅ Manejo eficiente de imágenes TIFF de alta resolución (>3 GB)
    ✅ Pipeline end-to-end desde imagen cruda hasta diagnóstico final
    ✅ Agregación inteligente con validación de criterios clínicos (mínimo 10 glomérulos)
    ✅ Visualizaciones detalladas y reportes médicos automatizados
    ✅ Configuración flexible mediante archivos YAML

🏗️ Arquitectura del Sistema

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Imagen TIFF   │───▶│  Detección YOLO  │───▶│  Glomérulos     │
│   Alta Resol.   │    │    (YOLOv8)      │    │   Detectados    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Diagnóstico   │◀───│   Agregación     │◀───│  Clustering    │
│     Final       │    │  (Voto Mayoria)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘

📊 Clases de Nefritis Lúpica

El sistema clasifica glomérulos en 3 clases agrupadas:
Clase 	Descripción 	Clases ISN/RPS Originales
Clase 0 	Mínima/Mesangial 	Clase I + II + V
Clase 1 	Proliferativa 	Clase III + IV
Clase 2 	Esclerosis 	Clase VI
Clase 3   Exclude
🚀 Instalación
Requisitos del Sistema

    Python 3.11 
    CUDA 11.8+ (recomendado para GPU)
    16 GB RAM mínimo (32 GB recomendado)
    50 GB espacio libre en disco

1. Clonar el Repositorio

git clone

2. Crear Entorno Virtual

# Conda (recomendado)
conda create -n lupus python=3.11
conda activate lupus

# O usando venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

3. Instalar Dependencias

# Instalar PyTorch (ajustar según tu sistema)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. Verificar Instalación

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())"


🔧 Configuración

El sistema se configura mediante el archivo config/config.yaml. Las secciones principales son:
Configuración de Detección

detection:
  model_name: "yolov8n"
  epochs: 100
  batch_size: 12
  confidence_threshold: 0.5
  input_size: 1536

Configuración de Clasificación


📚 Uso del Sistema
1. Preparación de Datos

Antes de entrenar, organiza y valida tus datos:

# Validar calidad de imágenes crudas
--check_pairs.py
--eleccion de tamano patches(1).py

# Tiling
-- winner_15.py

# Convertir anotaciones a formato YOLO
--conversor_geojson_to_yolo_mydataset.py

#Aumentación de datos offline
--augmentations_for_minorities.py

# Normalización de color (Reinhard)
-- stain_normalization.py

# Crear splits train/val/test
python src/split_patients.py \
  --images-dir data/raw/images \
  --annotations-dir data/annotations/yolo \
  --output-dir data/processed/detection


Salidas del entrenamiento:

    Modelo entrenado: models/detection/train_YYYYMMDD_HHMMSS/best.pt
    Métricas: models/detection/train_YYYYMMDD_HHMMSS/results.csv
    Gráficos: models/detection/train_YYYYMMDD_HHMMSS/training_curves.png


3. Diagnóstico Completo (Pipeline)
Procesar una imagen individual:

python src/br-038.ipynb \
  --detection-model models/detection/best.pt \
  --input data/biopsia_001.tiff \
  --output results/biopsia_001

📊 Interpretación de Resultados
Archivo de Diagnóstico (diagnosis.json)

{
  "image_path": "biopsia_001.tiff",
  "final_diagnosis": "Clase_Proliferativa",
  "confidence": 0.847,
  "valid_glomeruli": 15,
  "class_distribution": {
    "Clase_Minima_Mesangial": 2,
    "Clase_Proliferativa": 11,
    "Clase_Membranosa_Esclerosis": 2
  },
  "quality_metrics": {
    "meets_minimum_glomeruli": true,
    "avg_classification_confidence": 0.823
  }
}

Visualizaciones Generadas

    detections_overlay.jpg: Imagen con glomérulos detectados y etiquetados
    class_distribution.png: Gráfico de distribución de clases
    summary_report.png: Reporte visual completo
    glomeruli_crops/: Recortes individuales de cada glomérulo clasificado

Criterios de Validez

✅ Diagnóstico válido si:

    Se detectan ≥10 glomérulos
    Confianza promedio >0.8
    No hay artefactos significativos en la imagen

⚠️ Revisar manualmente si:

    5-9 glomérulos detectados
    Confianza 0.6-0.8
    Distribución muy equilibrada entre clases

❌ Diagnóstico no válido si:

    <5 glomérulos detectados
    Confianza <0.6
    Imagen con artefactos severos

🧪 Ejemplos de Uso
Ejemplo 1: Análisis Individual con API Python

from src.full_pipeline import LupusNephritisPipeline

# Inicializar pipeline
pipeline = LupusNephritisPipeline('config/config.yaml')

# Cargar modelos
pipeline.load_models(
    detection_model_path='models/detection/best.pt',
)

# Procesar biopsia
diagnosis = pipeline.process_biopsy(
    image_path='data/biopsia_ejemplo.tiff',
    save_results=True,
    output_dir=Path('results/ejemplo')
)

print(f"Diagnóstico: {diagnosis.final_diagnosis}")
print(f"Confianza: {diagnosis.confidence:.3f}")
print(f"Glomérulos válidos: {diagnosis.valid_glomeruli}")

Ejemplo 2: Notebook Jupyter Interactivo

# Cargar dependencias
%matplotlib inline
import matplotlib.pyplot as plt
from src.full_pipeline import LupusNephritisPipeline

# Configurar pipeline
pipeline = LupusNephritisPipeline()
pipeline.load_models('models/detection/best.pt')

# Procesar y visualizar
diagnosis = pipeline.process_biopsy('data/test_biopsy.tiff')

# Mostrar distribución de clases
plt.figure(figsize=(10, 6))
plt.bar(diagnosis.class_distribution.keys(), diagnosis.class_distribution.values())
plt.title(f'Diagnóstico: {diagnosis.final_diagnosis} (Confianza: {diagnosis.confidence:.3f})')
plt.show()

📈 Métricas de Rendimiento
Detección de Glomérulos (YOLOv8)

    mAP@0.5: 0.63
    mAP@0.5:0.95: 0.39
    Precision: 0.55
    Recall: 0.64
    Tiempo de inferencia: ~2.3s por imagen (GPU RTX 3080)


Pipeline Completo

    Accuracy diagnóstico biopsia: 0.64
    Sensibilidad por clase:
        Clase Mínima/Mesangial: 0.88
        Clase Proliferativa: 0.94
        Clase Membranosa/Esclerosis: 0.87
    Tiempo total: ~45s por biopsia (incluyendo visualizaciones)

🔬 Detalles Técnicos
Arquitecturas de Modelos
Detección: YOLOv8

    Backbone: CSPDarknet53
    Neck: PAN-FPN
    Head: Decoupled head con anchor-free detection
    Augmentations: Mosaic, MixUp, Geometric transforms
    Loss: Complete IoU Loss + Binary Cross Entropy



Procesamiento de Imágenes TIFF

class TIFFLoader:
    """Carga optimizada de imágenes TIFF grandes"""
    
    def load_tiff_lazy(self, image_path):
        # Lazy loading con zarr para imágenes >4GB
        store = tifffile.imread(image_path, aszarr=True)
        return zarr.open(store, mode='r')
    
    def get_tiles(self, image, tile_size=2048, overlap=256):
        # Procesamiento por tiles con solapamiento
        # Maneja imágenes de hasta 50,000×50,000 píxeles

Agregación de Diagnóstico

El sistema implementa múltiples estrategias de agregación:

    Voto por Mayoría: Clase más frecuente
    Promedio Ponderado: Por confianza de clasificación
    Validación de Criterios: Mínimo 10 glomérulos, distribución coherente

def aggregate_diagnosis(self, classifications):
    if len(classifications) < 10:
        return "INSUFICIENTES_GLOMERULOS"
    
    class_counts = Counter(c.class_name for c in classifications)
    final_class = class_counts.most_common(1)[0][0]
    confidence = class_counts[final_class] / len(classifications)
    
    return final_class, confidence

🐛 Solución de Problemas
Error: "CUDA out of memory"

detection:
  batch_size: 8   # Reducir de 16 a 8

Error: "No module named 'ultralytics'"

pip install ultralytics>=8.0.0

Error: Imágenes TIFF no cargan

# Verificar instalación de tifffile
pip install tifffile zarr

# Para imágenes muy grandes, usar lazy loading
python -c "from src.tiff_loader import TIFFLoader; loader = TIFFLoader(); print('TIFF loader OK')"

Rendimiento lento en CPU

# Verificar que PyTorch detecta GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Si no hay GPU, optimizar para CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

📋 Formato de Datos
Estructura Requerida para Entrenamiento
Detección (YOLO format):

data/detection/
├── images/
│   ├── train/
│   │   ├── biopsy_001.tiff
│   │   └── biopsy_002.tiff
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── biopsy_001.txt    # class x_center y_center width height
    │   └── biopsy_002.txt    # 0 0.5 0.3 0.1 0.08
    ├── val/
    └── test/


Conversión de Formatos

El sistema soporta conversión entre múltiples formatos:

# Pascal VOC → YOLO
python src/prepare_data_script.py \
  --convert-annotations data/annotations/pascal_voc \
  --annotation-format yolo

# COCO → YOLO
python src/prepare_data_script.py \
  --convert-annotations data/annotations/coco.json \
  --annotation-format yolo

Áreas de Mejora

    Implementar segmentación semántica además de detección
    Agregar más arquitecturas de clasificación (ConvNeXt, Swin Transformer)
    Optimizar para deployment en edge devices
    Integrar con sistemas PACS hospitalarios
    Desarrollar interfaz web interactiva
    Añadir soporte para más tipos de nefritis

📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para detalles.

🩺 Mejorando el diagnóstico de nefritis lúpica mediante IA 🔬

⬆ Volver arriba
