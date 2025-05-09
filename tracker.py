import cv2
import torch
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("runs/detect/train2/weights/best.pt")

# Abrir el video
video_path = "canasta_3D_acierto_tirolibre.mp4"
cap = cv2.VideoCapture(video_path)

# Obtener información del video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Configurar el writer para guardar el video procesado
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Procesar el video frame por frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Terminar si no hay más frames

    # Realizar tracking en el frame
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="cuda", conf=0.2, iou=0.2, max_det=1, verbose=False)

    # Obtener detecciones y modificar el frame
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja

                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostrar el frame modificado en tiempo real
    cv2.imshow("Tracking", frame)

    # Guardar el frame modificado
    out.write(frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
