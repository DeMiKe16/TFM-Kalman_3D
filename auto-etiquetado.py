import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from datetime import datetime

# Configuración del modelo
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Carpeta de imágenes y etiqueta a detectar
image_folder = "dataset_canasta3D"
text = "basket ball"

# Estructura para COCO JSON
coco_data = {
    "info": {
        "year": datetime.now().year,
        "version": "1.0",
        "description": "Auto-labeled dataset",
        "contributor": "GroundingDINO",
        "date_created": datetime.now().isoformat()
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "name": "Public Domain"
        }
    ],
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "ball", "supercategory": "object"}]
}

image_list = os.listdir(image_folder)
total_images = len(image_list)
annotation_id = 1  # ID único para cada anotación

# Procesar cada imagen en la carpeta
for idx, image_name in enumerate(image_list, start=1):
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.2,  # Reducido para capturar más detecciones
        text_threshold=0.2,
        target_sizes=[image.size[::-1]]
    )
    
    # Agregar imagen a COCO JSON
    coco_data["images"].append({
        "id": idx,
        "license": 1,
        "file_name": image_name,
        "width": width,
        "height": height,
        "date_captured": datetime.now().isoformat()
    })
    
    # Procesar detecciones
    for det in results:
        for score, box in zip(det['scores'], det['boxes']):
            x_min, y_min, x_max, y_max = box.tolist()
            bbox = [
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min
            ]
            area = bbox[2] * bbox[3]  # Ancho * Alto
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": idx,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "segmentation": [],
                "score": float(score),
                "iscrowd": 0
            })
            annotation_id += 1  # Incrementar ID de anotación
    
    # Mostrar progreso
    print(f"Procesado {idx}/{total_images}: {image_name}")

# Guardar etiquetas en formato COCO JSON
output_json = os.path.join(image_folder, "labels_coco.json")
with open(output_json, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"Etiquetado completado. Resultados guardados en {output_json}")
