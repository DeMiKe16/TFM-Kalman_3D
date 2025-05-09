from moviepy.video.io.VideoFileClip import VideoFileClip

# Cargar el video original
video_path = "3dcanastasnuevo.mp4"
clip = VideoFileClip(video_path)

# Recortar del segundo 30 al 35
subclip = clip.subclipped(155, 159)

# Guardar el video recortado
output_path = "canasta_3D_fallo_triple.mp4"
subclip.write_videofile(output_path, codec="libx264", fps=clip.fps)

print(f"Video recortado guardado en: {output_path}")
