from PIL import Image
img = Image.open("biopsia_convertida_10.png")
print(img.size)  # tamaño total W,H
# luego mides manualmente con anotaciones o bounding boxes
