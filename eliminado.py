import json
import os

# Cargar el archivo labels_coco.json
image_folder = "dataset_canasta3D"
input_json = os.path.join(image_folder, "labels_coco.json")
output_json = os.path.join(image_folder, "labels_coco_cleaned.json")

with open(input_json, "r") as f:
    coco_data = json.load(f)

# Función para calcular la intersección sobre unión (IoU) entre dos cajas
def iou(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Filtrar duplicados dentro de cada imagen
iou_threshold = 0.5  # Umbral para considerar dos detecciones como duplicadas
filtered_annotations = []
seen_annotations = {}
removed_count = 0

for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]

    if image_id not in seen_annotations:
        seen_annotations[image_id] = []

    # Verificar si la bbox es similar a otra ya guardada
    is_duplicate = any(iou(bbox, existing_bbox) > iou_threshold for existing_bbox in seen_annotations[image_id])

    if not is_duplicate:
        seen_annotations[image_id].append(bbox)
        filtered_annotations.append(annotation)
    else:
        removed_count += 1

# Guardar el nuevo JSON sin duplicados
coco_data["annotations"] = filtered_annotations

with open(output_json, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"Proceso completado. Etiquetas filtradas guardadas en {output_json}")
print(f"Se han eliminado {removed_count} etiquetas duplicadas.")
