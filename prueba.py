import cv2
import numpy as np

# Lista para almacenar los puntos seleccionados
puntos = []

# Función para capturar los puntos seleccionados con el mouse
def seleccionar_punto(event, x, y, flags, param):
    global puntos, frame_resized

    # Si se hace clic izquierdo, añadir el punto a la lista
    if event == cv2.EVENT_LBUTTONDOWN:
        x_original = int(x / scale_factor)
        y_original = int(y / scale_factor)
        puntos.append((x_original, y_original))

        # Dibujar un círculo en el punto seleccionado
        cv2.circle(frame_resized, (x, y), 5, (0, 0, 255), -1)

        # Mostrar el punto en la ventana
        cv2.imshow("Seleccionar puntos", frame_resized)

# Cargar el video
cap = cv2.VideoCapture('canasta_3D_acierto_tirolibre.mp4')

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Leer el primer frame
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo leer el primer frame.")
    cap.release()
    exit()

# Redimensionar el frame si es demasiado grande
screen_height = 720  # Puedes ajustar este valor según tu pantalla
scale_factor = screen_height / frame.shape[0] if frame.shape[0] > screen_height else 1
frame_resized = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))

# Hacer la ventana redimensionable
cv2.namedWindow("Seleccionar puntos", cv2.WINDOW_NORMAL)
cv2.imshow("Seleccionar puntos", frame_resized)
cv2.setMouseCallback("Seleccionar puntos", seleccionar_punto)

# Esperar hasta que el usuario presione 'q' para salir
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir al presionar 'q'
        break

# Imprimir las coordenadas de los puntos seleccionados
print("Puntos seleccionados:", puntos)

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
