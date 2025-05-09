import cv2

# Cargar el video
video_path = 'canasta_3D_acierto_tirolibre.mp4'  # Sustituye esto con la ruta de tu video
cap = cv2.VideoCapture(video_path)

# Verificar si el video se ha cargado correctamente
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

while True:
    # Leer un frame
    ret, frame = cap.read()

    # Si no se puede leer el frame, salir
    if not ret:
        print("Fin del video")
        break

    # Mostrar el frame
    cv2.imshow("Frame", frame)

    # Esperar por una tecla
    key = cv2.waitKey(0)  # Espera hasta que se presione una tecla
    if key == 27:  # Si presionas 'Esc', salir
        break
    elif key == ord('n'):  # Si presionas 'n', avanzar al siguiente frame
        continue

# Liberar recursos y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
