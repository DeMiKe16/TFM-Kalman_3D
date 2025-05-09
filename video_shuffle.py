import cv2
import os
import random
import shutil

def extract_and_shuffle_frames(video_path, output_folder="dataset_canasta3D"):
    # Crear la carpeta de salida si no existe
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Capturar el video
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    frame_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_index:06d}.png")
        frame_list.append(frame_filename)
        cv2.imwrite(frame_filename, frame)
        frame_index += 1
    
    cap.release()
    
    # Barajar los frames
    random.shuffle(frame_list)
    
    # Renombrar los archivos de forma aleatoria
    for i, frame_path in enumerate(frame_list):
        new_name = os.path.join(output_folder, f"{i:06d}.png")
        os.rename(frame_path, new_name)
        
        # Mostrar progreso
        print(f"Procesado {i}/{frame_index}: {new_name}")
    
    print(f"Proceso completado. Frames guardados en {output_folder}")

# Uso
video_input = "entrenamiento_3dcanastas.mp4"  # Cambia esto por el path de tu video
extract_and_shuffle_frames(video_input)