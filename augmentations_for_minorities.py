import albumentations as A
import cv2
import os
import random
import numpy as np  # Importar numpy para asegurarse de que las imágenes sean arrays

# Definir las transformaciones de aumentación (más suaves)
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),  # Flip horizontal sin cambiar el brillo, etc.
    A.RandomCrop(width=1536, height=1536, p=1),  # Recorte aleatorio
])

def load_image_and_label(image_path, label_path):
    """
    Carga la imagen y la etiqueta correspondiente.
    """
    # Leer la imagen con OpenCV
    image = cv2.imread(image_path)  # Cargar la imagen
    if image is None:  # Verificar si la imagen se cargó correctamente
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Verificar si la imagen está en el formato adecuado de numpy.ndarray
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Imagen no es un numpy array: {image_path}")

    label = []
    
    # Cargar el archivo de etiquetas en formato YOLO
    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            cls_id = int(parts[0])  # 0: no proliferativo, 1: proliferativo, 2: esclerosado, 3: exclude
            x_center, y_center, w, h = map(float, parts[1:])
            label.append([cls_id, x_center, y_center, w, h])  # [class_id, x_center, y_center, width, height]
    
    return image, label

def augment_image(image, label):
    """
    Aplica las transformaciones a una imagen y su máscara (etiquetas).
    """
    augmented = transform(image=image)
    augmented_image = augmented['image']
    
    # Aumentación de las etiquetas (cambiar coordenadas para cada transformación)
    augmented_labels = []
    for obj in label:
        cls_id, x_center, y_center, w, h = obj
        augmented_labels.append([cls_id, x_center, y_center, w, h])

    return augmented_image, augmented_labels

def generate_augmented_images(images, labels, target_count, class_ids):
    augmented_images = []
    augmented_labels = []
    
    while len(augmented_images) < target_count:
        idx = random.randint(0, len(images)-1)  # Elegir aleatoriamente una imagen y su etiqueta
        image = images[idx]
        label = labels[idx]
        
        # Solo aumentar imágenes de las clases especificadas (esclerosado = 2, exclude = 3)
        if any(cls_id in class_ids for cls_id, _, _, _, _ in label):  # Filtrar por las clases que quieres aumentar
            augmented_image, augmented_label = augment_image(image, label)
            augmented_images.append(augmented_image)
            augmented_labels.append(augmented_label)
    
    return augmented_images, augmented_labels

def save_augmented_images_and_labels(augmented_images, augmented_labels, save_path, original_img_name, start_idx):
    """
    Guarda las imágenes aumentadas y sus etiquetas correspondientes.
    - Mantiene el nombre del tile original pero agrega un sufijo o índice para diferenciarlas.
    """
    # Directorios para las imágenes y las etiquetas
    images_save_path = os.path.join(save_path, "images")
    labels_save_path = os.path.join(save_path, "labels")
    
    # Crear las carpetas si no existen
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)

    for idx, (aug_img, aug_label) in enumerate(zip(augmented_images, augmented_labels)):
        # Aquí mantenemos el nombre original de la imagen y le agregamos un índice único
        img_name = f"{original_img_name}_{start_idx + idx}.png"  # Ejemplo: tile_1_0.png, tile_1_1.png
        label_name = f"{original_img_name}_{start_idx + idx}.txt"  # Etiqueta correspondiente
        
        # Guardar imagen en la carpeta "images"
        cv2.imwrite(os.path.join(images_save_path, img_name), aug_img)
        
        # Guardar etiquetas en la carpeta "labels" en formato YOLO
        with open(os.path.join(labels_save_path, label_name), 'w') as label_file:
            for obj in aug_label:
                cls_id, x_center, y_center, w, h = obj
                label_file.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")

# Directorios de entrada
image_dir = "C:/Users/mayca/Downloads/archive/dataset_parcial_prueba_para_winner_15/images"  # Ruta a las imágenes
label_dir = "C:/Users/mayca/Downloads/archive/dataset_parcial_prueba_para_winner_15/labels"  # Ruta a las etiquetas (carpeta "labels")

# Cargar todas las imágenes y etiquetas
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
label_paths = [os.path.join(label_dir, fname.replace('.png', '.txt')) for fname in os.listdir(image_dir) if fname.endswith('.png')]

images = []
labels = []

# Cargar todas las imágenes y etiquetas
for image_path, label_path in zip(image_paths, label_paths):
    image, label = load_image_and_label(image_path, label_path)
    images.append(image)
    labels.append(label)

# Contar el número de imágenes de las clases minoritarias (esclerosado = 2 y exclude = 3)
minority_esclerosado = [label for label in labels if any(cls_id == 2 for cls_id, _, _, _, _ in label)]
minority_exclude = [label for label in labels if any(cls_id == 3 for cls_id, _, _, _, _ in label)]

# Calcular un aumento del 30% para cada clase minoritaria
augmentation_percentage = 0.30
target_count_esclerosado = int(len(minority_esclerosado) * augmentation_percentage)  # Aumento del 30%
target_count_exclude = int(len(minority_exclude) * augmentation_percentage)  # Aumento del 30%

# Definir las clases a las que se les realizará el oversampling (esclerosado = 2, exclude = 3)
class_ids = [2, 3]

# Generar imágenes aumentadas para las clases minoritarias
augmented_esclerosado, augmented_labels_esclerosado = generate_augmented_images(minority_esclerosado, labels, target_count_esclerosado, class_ids)
augmented_exclude, augmented_labels_exclude = generate_augmented_images(minority_exclude, labels, target_count_exclude, class_ids)

# Ruta donde guardar las imágenes y etiquetas aumentadas
save_path = "C:/Users/mayca/OneDrive/Escritorio/augmentations"  # Cambia esto al directorio donde quieres guardar los datos aumentados

# Guardar las imágenes y etiquetas aumentadas para "esclerosado"
for idx, image in enumerate(minority_esclerosado):
    original_img_name = os.path.basename(image_paths[idx]).replace(".png", "")
    save_augmented_images_and_labels(augmented_esclerosado, augmented_labels_esclerosado, save_path, original_img_name, 0)

# Guardar las imágenes y etiquetas aumentadas para "exclude"
for idx, image in enumerate(minority_exclude):
    original_img_name = os.path.basename(image_paths[idx]).replace(".png", "")
    save_augmented_images_and_labels(augmented_exclude, augmented_labels_exclude, save_path, original_img_name, target_count_esclerosado)






