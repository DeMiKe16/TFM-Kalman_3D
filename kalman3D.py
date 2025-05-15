import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from umucv.util import cameraOutline
from umucv.kalman import ukf
from umucv.htrans import sepcam


# ---------------------- FUNCIONES AUXILIARES -----------------------
# Funciones de transformación homogénea optimizadas
def homog(x):
    """Convierte coordenadas a coordenadas homogéneas añadiendo 1."""
    ax = np.asarray(x)
    return np.concatenate([ax, np.ones(ax.shape[:-1] + (1,))], axis=-1)

def inhomog(x):
    """Convierte coordenadas homogéneas a inhomogéneas."""
    ax = np.asarray(x)
    return ax[..., :-1] / ax[..., [-1]]

def htrans(h, x):
    """Aplicar transformación homogénea h a los puntos x."""
    return inhomog(homog(x) @ h.T)

def rmsreproj(view, model, transf):
    """Calcular error cuadrático medio de reproyección."""
    err = view - htrans(transf, model)
    return np.sqrt(np.mean(np.square(err)))

# Función para calcular la pose inicial
def pose(K, image, model):
    """Calcula la matriz de transformación a partir de puntos 2D y 3D."""
    ok, rvec, tvec = cv.solvePnP(model, image, K, np.zeros(4))
    if not ok:
        return 1e6, None
        
    R, _ = cv.Rodrigues(rvec)
    M = K @ np.hstack((R, tvec))
    rms = rmsreproj(image, model, M)
    return rms, M

# Crear el aro
def crear_aro(centro_x=0, centro_y=12.425, centro_z=3.05, radio=0.225, num_puntos=16):
    """Crea puntos para representar un aro de baloncesto."""
    angulos = np.linspace(0, 2*np.pi, num_puntos, endpoint=False)
    x = centro_x + radio * np.cos(angulos)
    y = centro_y + radio * np.sin(angulos)
    z = np.full_like(x, centro_z)
    
    # Convertir a array y añadir el primer punto al final para cerrar el círculo
    aro = np.column_stack((x, y, z))
    return np.vstack((aro, aro[0]))

# Dibujar una polilínea en 3D
def plot3(ax, c, color):
    """Dibuja una curva 3D en el eje dado."""
    x, y, z = c.T
    ax.plot(x, y, z, color)

# Ajustar la escala de los ejes para que se vean proporcionales
def set_axes_equal(ax):
    """Ajusta los ejes para que tengan la misma escala."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([max(origin[2] - radius, 0), origin[2] + radius])  # Limite inferior Z en 0

# Manejadores de eventos para las teclas
def setup_event_handlers(fig):
    """Configura los manejadores de eventos para las teclas."""
    state = {'fin': False, 'kalman_active': False}
    
    def on_press(event):
        if event.key == 'escape':
            state['fin'] = True
        elif event.key == 'm':
            state['kalman_active'] = True
            
    fig.canvas.mpl_connect('key_press_event', on_press)
    return state

def calcular_posicion_pelota(K, R, C, coords_imagen, profundidad):
    """
    Calcula la posición 3D de una pelota en coordenadas del mundo.
    
    Parámetros:
    - K: Matriz de calibración intrínseca de la cámara (3x3)
    - R: Matriz de rotación de la cámara (3x3)
    - C: Centro de la cámara en coordenadas del mundo (vector 3x1)
    - coords_imagen: Coordenadas de la pelota en la imagen [u, v]
    - profundidad: Profundidad real de la pelota
    
    Retorna:
    - Posición 3D de la pelota en coordenadas del mundo
    """
    # Convertir coordenadas de imagen a coordenadas homogéneas
    u, v = coords_imagen
    coords_homogeneas = np.array([u, v, 1])
    
    # Calcular coordenadas normalizadas usando la inversa de K
    K_inv = np.linalg.inv(K)
    coords_normalizadas = K_inv @ coords_homogeneas
    
    # Escalar el vector por la profundidad
    direccion_camara = coords_normalizadas * profundidad
    
    # Convertir de coordenadas de la cámara a coordenadas del mundo
    # Usando R transpuesta (R.T) porque necesitamos la transformación inversa
    posicion_mundo = C + R.T @ direccion_camara
    
    return posicion_mundo

def estimar_velocidad_inicial(p0, pf, theta_deg=45):
    """
    Calcula vector velocidad inicial con módulo igual a distancia 3D
    entre p0 y pf, y ángulo de tiro theta (altura en Z).
    
    p0, pf: posición inicial y final (x,y,z)
    theta_deg: ángulo de tiro (por defecto 45º)
    
    Retorna:
    vector velocidad inicial (vx, vy, vz)
    """
    x0, y0, z0 = p0
    xf, yf, zf = pf

    # Distancia 3D
    distancia = np.linalg.norm(pf - p0) * 1.75
    
    theta = np.radians(theta_deg)
    
    # Vector horizontal XY normalizado
    d_xy = np.array([xf - x0, yf - y0])
    d_xy_norm = np.linalg.norm(d_xy)
    if d_xy_norm == 0:
        raise ValueError("La posición inicial y final son iguales en el plano horizontal XY.")
    dir_xy = d_xy / d_xy_norm
    
    # Componentes horizontal y vertical de la velocidad
    v_h = distancia * np.cos(theta)
    v_z = distancia * np.sin(theta)
    
    # Componentes vx, vy en el plano horizontal
    v_x, v_y = v_h * dir_xy
    
    return np.array([v_x, v_y, v_z])

def main():
    
    # ---------------------- INICIALIZACIÓN -----------------------
    # Matriz de calibración
           
    data = np.load('calibracion_gran_angular.npz')

    # Extraer los parámetros (ajusta los nombres a lo que hayas guardado)
    K = data['mtx']  # o data['camera_matrix']
    D_matrix = data['dist']  # o data['dist_coeff']
    
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]

    # Coordenadas de referencia en la imagen (en píxeles)
    referencias_imagen = np.array([
        [691, 286],
        [886, 316],
        [886, 454],
        [691, 436],
        [1272, 766],
        [1216, 775],
        [928, 813],
        [1594, 909],
        [1050, 1074],
        [384, 889],
        [1378, 976],
        [1758, 1047]  
    ], dtype='float32')

    # Puntos del modelo en 3D
    referencias_real = np.array([
        [-0.9, 12.8, 3.95],
        [0.9, 12.8, 3.95],
        [0.9, 12.8, 2.9],
        [-0.9, 12.8, 2.9],
        [7.5, 14, 0],
        [6.6, 14, 0],
        [2.45, 14, 0],
        [2.45, 8.2, 0],
        [-2.45, 8.2, 0],
        [-2.45, 14, 0],
        [0, 8.2, 0],
        [0, 6.4, 0]
    ], dtype='float32')

    # Definir geometrías de la cancha
    canasta = np.array([
        [-0.9, 12.8, 3.95],
        [0.9, 12.8, 3.95],
        [0.9, 12.8, 2.9],
        [-0.9, 12.8, 2.9],
        [-0.9, 12.8, 3.95]
    ], dtype='float32')

    campo = np.array([
        [7.5, 14, 0],
        [-7.5, 14, 0]
    ], dtype='float32')  

    cuadrado = np.array([
        [2.45, 14, 0],
        [2.45, 8.2, 0],
        [-2.45, 8.2, 0],
        [-2.45, 14, 0],
        [2.45, 14, 0]
    ], dtype='float32')    
             
    # Calcular la pose solo al principio
    err, Me = pose(K, referencias_imagen, referencias_real)
    print(f"Error inicial: {err}")
    print(f"Matriz de transformación: \n{Me}")
    
    # Parámetros del aro
    centro_x = 0
    centro_y = 12.425
    centro_z = 3.05
    radio_aro = 0.225

    # Crear el aro y proyectar geometrías a 2D
    aro = crear_aro()
    canasta_view_2D = htrans(Me, canasta).astype(int)
    aro_2d = htrans(Me, aro).astype(int)
    cuadrado_2d = htrans(Me, cuadrado).astype(int)

    # Usar la matriz de transformación para obtener el contorno de la cámara
    camline = cameraOutline(Me)
    
    K_Me, R_Me, C_Me = sepcam(Me)
    print(f"K: \n{K_Me}")
    print(f"R: \n{R_Me}")
    print(f"C: \n{C_Me}")

    # Configurar la figura de Matplotlib
    plt.ion()
    fig = plt.figure(figsize=(19.2, 10.8))
    state = setup_event_handlers(fig)

    # Subplot para el video
    ax1 = fig.add_axes([0.02, 0.3, 0.5, 0.6])
    ax1.set_title("Video")
    ax1.axis('off')
    im = ax1.imshow(np.zeros((1080, 1920, 3), dtype=np.uint8), aspect='auto')

    # Subplot para la gráfica 3D
    ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8], projection='3d')
    ax2.set_title("Gráfica 3D")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Dibujar el escenario 3D inicial
    plot3(ax2, canasta, 'r')  # Tablero en rojo
    plot3(ax2, aro, 'orange')  # Aro en naranja
    plot3(ax2, campo, 'g')  # Fondo en verde
    plot3(ax2, cuadrado, 'g')  # Cuadrado en verde
    plot3(ax2, camline, 'b')  # Línea de la cámara en azul
    set_axes_equal(ax2)  # Ajustar la escala de los ejes

    # Cargar el modelo YOLO entrenado
    model = YOLO("runs/detect/train2/weights/best.pt")

    # Configuración de kalman
    fps = 62  # 63 para triple 
    dt = 1/61.5
    a = np.array([0, 0, -9.8])  # Aceleración (gravedad)
    
    d_real = 0.24 

    # ---------------------- KALMAN -----------------------
    # Matriz de transición de estado F
    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])

    # Matriz de control B
    B = np.array([
        [dt**2/2, 0,       0      ],
        [0,       dt**2/2, 0      ],
        [0,       0,       dt**2/2],
        [dt,      0,       0      ],
        [0,       dt,      0      ],
        [0,       0,       dt     ]
    ])

    # Funciones para el filtro UKF
    def f(x):
        """Función de transición de estado."""
        return F@x + B@a

    def h(x):
        """Función de medición: proyecta puntos 3D a 2D."""
        return htrans(Me, x[:3])

    def b(x):
        """Función de control (no usada aquí)."""
        return 0

    # Estado inicial y covarianza
    mu = np.array([0, 0, 0, 0, 5, 5])  # Posición y velocidad inicial [-2.45, -5.8, 2, 0, 4, 5] -> tiro libre
    P = np.diag([1, 1, 1, 4, 4, 4])**2  # Covarianza inicial

    # Matrices de ruido
    sigmaM = 1e-3  # Ruido del modelo
    sigmaZ = 20       # Ruido de medición
    Q = sigmaM**2 * np.eye(6)  # Covarianza del ruido del proceso
    R = sigmaZ**2 * np.eye(2)  # Covarianza del ruido de medición

    # Listas para almacenar resultados
    res = []
    listCenterX = []
    listCenterY = []
    listpuntos = []
    
    imagen = 0

    # Abrir el video
    video_path = "canasta_3D_acierto_tirolibre.mp4"
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        return

    # ---------------------- BUCLE PRINCIPAL -----------------------
    # Procesar el video frame por frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Terminar si no hay más frames

        center_x = None
        center_y = None

        # Realizar tracking en el frame
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            device="cuda", 
            conf=0.2, 
            iou=0.2, 
            max_det=1, 
            verbose=False
        )

        # Obtener detecciones y modificar el frame
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
                    
                    # Calcular el centro de la detección
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Dibujar un círculo en el centro de la detección
                    cv.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
                    
        if imagen == 85: # 85 para acierto en tiro libre, 83 para fallo en tiro libre, 65 para acierto en triple, 55 para fallo en triple
            state['kalman_active'] = True
            dim_imagen = max(x2 - x1, y2 - y1) / 1.25
            z_real = (f_x * d_real) / dim_imagen
            # Calcular la posición 3D de la pelota en el mundo
            posicion_real = calcular_posicion_pelota(K_Me, R_Me, C_Me, [center_x, center_y], z_real)
            print("Posición real de la pelota en el espacio 3D:", posicion_real)
            velocidad_real = estimar_velocidad_inicial(posicion_real, np.array([centro_x, centro_y, centro_z]), 45)
            print("Velocidades iniciales necesarias:", velocidad_real)
            mu[0:3] = posicion_real
            mu[3:6] = velocidad_real
            
        # Dibujar elementos en el frame
        # Dibujar el tablero
        for i in range(len(canasta_view_2D) - 1):
            cv.line(frame, tuple(canasta_view_2D[i]), tuple(canasta_view_2D[i + 1]), (0, 0, 255), 2)
            
        # Dibujar puntos de referencia
        for i, point in enumerate(referencias_imagen):
            cv.circle(frame, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
        
        # Dibujar el aro
        for i in range(len(aro_2d) - 1):
            cv.line(frame, tuple(aro_2d[i]), tuple(aro_2d[i + 1]), (0, 0, 255), 2)
        
        for i in range(len(cuadrado_2d) - 1):
            cv.line(frame, tuple(cuadrado_2d[i]), tuple(cuadrado_2d[i + 1]), (0, 255, 0), 2)
        
        
        # ---------------------- BUCLE KALMAN -----------------------
        # Actualizar el filtro de Kalman si está activo
        if state['kalman_active']:
            # Actualizar el filtro con o sin medición
            measurement = np.array([center_x, center_y]) if center_x is not None else None
            mu, P, pred = ukf(mu, P, f, Q, b, a, measurement, h, R)
            print(f"mu: {mu}")
            
            # Guardar resultados
            res.append((mu, P))
            
            if center_x is not None:
                listCenterX.append(center_x)
                listCenterY.append(center_y)
                listpuntos.append((center_x, center_y, "normal"))
            else:
                listpuntos.append((None, None, "None"))
            
            # Proyectar puntos históricos
            xe = [m[0] for m, _ in res]
            xu = [2*np.sqrt(p[0,0]) for _, p in res]
            ye = [m[1] for m, _ in res]
            yu = [2*np.sqrt(p[1,1]) for _, p in res] 
            ze = [m[2] for m, _ in res]
            zu = [2*np.sqrt(p[2,2]) for _, p in res]
            
            # Dibujar centros detectados
            for cx, cy in zip(listCenterX, listCenterY):
                cv.circle(frame, (int(cx), int(cy)), 3, (0, 255, 0), -1)
                
            # Proyectar y dibujar puntos estimados de la trayectoria
            points3D = np.array(list(zip(xe, ye, ze)), dtype=np.float32)
            points2D = htrans(Me, points3D)

            for n, (u, v) in enumerate(points2D):
                u, v = int(u), int(v)
                incertidumbre = int((abs(xu[n]) + abs(yu[n]) + abs(zu[n])) / 3)
                cv.circle(frame, (u, v), max(1, incertidumbre), (255, 255, 0), 1)
            
        # Mostrar el frame modificado en el subplot del video
        im.set_data(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # Limpiar la gráfica 3D antes de redibujar
        ax2.clear()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Redibujar el escenario 3D
        plot3(ax2, canasta, 'r')
        plot3(ax2, aro, 'orange')
        plot3(ax2, campo, 'g')
        plot3(ax2, cuadrado, 'g')
        plot3(ax2, camline, 'b')
        
        # Dibujar la trayectoria estimada si el filtro de Kalman está activo
        if state['kalman_active'] and res:
            mu_filtered = np.array([mu[:3] for mu, _ in res])
            if len(mu_filtered) > 0:
                plot3(ax2, mu_filtered, 'g')
                
        set_axes_equal(ax2)

        # Redibujar la figura y procesar eventos
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.001)
        
        imagen += 1

        # Salir si se presiona Escape
        if state['fin']:
            break

    # Liberar recursos
    cap.release()
    cv.destroyAllWindows()
    plt.ioff()
    plt.show()

# Ejecutar el programa si se ejecuta como script principal
if __name__ == "__main__":
    main()