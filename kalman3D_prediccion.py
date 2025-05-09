import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from umucv.util import cameraOutline, putText
from umucv.kalman import ukf


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
def crear_aro(centro_x=-2.45, centro_y=-1.576, centro_z=3.05, radio=0.225, num_puntos=16):
    """Crea puntos para representar un aro de baloncesto."""
    angulos = np.linspace(0, 2*np.pi, num_puntos, endpoint=False)
    x = centro_x + radio * np.cos(angulos)
    y = centro_y + radio * np.sin(angulos)
    z = np.full_like(x, centro_z)
    
    # Convertir a array y añadir el primer punto al final para cerrar el círculo
    aro = np.column_stack((x, y, z))
    return np.vstack((aro, aro[0]))

# Simular múltiples trayectorias
def simular_trayectorias(mu, P, num_parabolas=100, gravedad=-9.81, t=np.linspace(0, 1, 100)):
    # Generar trayectorias en un solo paso vectorizado
    vx0 = np.random.normal(mu[3], np.sqrt(P[3, 3]), num_parabolas)  # Velocidades iniciales x
    vy0 = np.random.normal(mu[4], np.sqrt(P[4, 4]), num_parabolas)  # Velocidades iniciales y
    vz0 = np.random.normal(mu[5], np.sqrt(P[5, 5]), num_parabolas)  # Velocidades iniciales z
    
    # Simular trayectorias
    x_sim = mu[0] + vx0[:, None] * t  # Esto genera un array (num_parabolas, tiempo_sim)
    y_sim = mu[1] + vy0[:, None] * t  # Efecto de la gravedad
    z_sim = mu[2] + vz0[:, None] * t + 0.5 * gravedad * t**2 # Movimiento en Z (gravedad o velocidad inicial)
    
    return x_sim, y_sim, z_sim

# Verificar intersección con el aro
def verificar_interseccion(x_sim, y_sim, z_sim, puntos_aro, radio_aro):
    intersecciones = []
    x_aro, y_aro, z_aro = puntos_aro[:, 0], puntos_aro[:, 1], puntos_aro[:, 2]
    
    for i in range(x_sim.shape[0]):  # Recorrer trayectorias
        for j in range(x_sim.shape[1]):  # Recorrer puntos en el tiempo
            distancia = np.sqrt(
                (x_sim[i, j] - x_aro)**2 +
                (y_sim[i, j] - y_aro)**2 +
                (z_sim[i, j] - z_aro)**2
            )
            if np.any(distancia <= radio_aro):
                intersecciones.append(i)
                break
    return intersecciones

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
    impresion = {'print': False}
    
    def on_press(event):
        if event.key == 'escape':
            state['fin'] = True
        elif event.key == 'm':
            impresion['print'] = True
            state['kalman_active'] = True
            
            
    fig.canvas.mpl_connect('key_press_event', on_press)
    return state, impresion

def main():
    
    # ---------------------- INICIALIZACIÓN -----------------------
    # Matriz de calibración
    K = np.array([[1666, 0, 969], 
                [0, 1666, 544], 
                [0, 0, 1]])

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
        [-3.35, -1.2, 3.95],
        [-1.55, -1.2, 3.95],
        [-1.55, -1.2, 2.9],
        [-3.35, -1.2, 2.9],
        [5.05, 0, 0],
        [4.15, 0, 0],
        [0, 0, 0],
        [0, -5.8, 0],
        [-4.9, -5.8, 0],
        [-4.9, 0, 0],
        [-2.45, -5.8, 0],
        [-2.45, -7.6, 0]
    ], dtype='float32')

    # Definir geometrías de la cancha
    canasta = np.array([
        [-3.35, -1.2, 3.95],
        [-1.55, -1.2, 3.95], 
        [-1.55, -1.2, 2.9],
        [-3.35, -1.2, 2.9],
        [-3.35, -1.2, 3.95]
    ], dtype='float32')

    fondo = np.array([
        [5.05, 0, 0],
        [-9.95, 0, 0]
    ], dtype='float32')  

    cuadrado = np.array([
        [0, 0, 0],
        [0, -5.8, 0],
        [-4.9, -5.8, 0],
        [-4.9, 0, 0],
        [0, 0, 0]
    ], dtype='float32')    
             
    # Calcular la pose solo al principio
    err, Me = pose(K, referencias_imagen, referencias_real)
    print(f"Error inicial: {err}")
    print(f"Matriz de transformación: \n{Me}")
    
    # Parámetros del aro
    centro_x = -2.45
    centro_y = -1.576
    centro_z = 3.05
    radio_aro = 0.225

    # Crear el aro y proyectar geometrías a 2D
    aro = crear_aro()
    canasta_view_2D = htrans(Me, canasta).astype(int)
    aro_2d = htrans(Me, aro).astype(int)
    fondo_2d = htrans(Me, fondo).astype(int)
    cuadrado_2d = htrans(Me, cuadrado).astype(int)

    # Usar la matriz de transformación para obtener el contorno de la cámara
    camline = cameraOutline(Me)

    # Configurar la figura de Matplotlib
    plt.ion()
    fig = plt.figure(figsize=(19.2, 10.8))
    state, impresion = setup_event_handlers(fig)

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
    plot3(ax2, fondo, 'g')  # Fondo en verde
    plot3(ax2, cuadrado, 'g')  # Cuadrado en verde
    plot3(ax2, camline, 'b')  # Línea de la cámara en azul
    set_axes_equal(ax2)  # Ajustar la escala de los ejes

    # Cargar el modelo YOLO entrenado
    model = YOLO("runs/detect/train2/weights/best.pt")

    # Configuración de kalman
    fps = 62  # 63 para triple 
    dt = 1/61.5
    t = np.arange(0, 2 + dt, dt)  # Vector de tiempo
    a = np.array([0, 0, -9.8])  # Aceleración (gravedad)

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
    mu = np.array([-3, -6.2, 2, 0, 4, 4])  # Posición y velocidad inicial
    P = np.diag([1, 1, 1, 1, 1, 1])**2  # Covarianza inicial

    # Matrices de ruido
    sigmaM = 0.00001  # Ruido del modelo
    sigmaZ = 4       # Ruido de medición
    Q = sigmaM**2 * np.eye(6)  # Covarianza del ruido del proceso
    R = sigmaZ**2 * np.eye(2)  # Covarianza del ruido de medición

    # Listas para almacenar resultados
    res = []
    listCenterX = []
    listCenterY = []
    listpuntos = []
    
    # Variables para el análisis de trayectorias
    calcular_probabilidad = True
    max_probabilidad = 0
    probabilidad_mostrar = 0
    frame_counter = 0

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
                    
        # Dibujar elementos en el frame
        # Mostrar la probabilidad en el frame
        putText(frame, f"Prob. canasta: {probabilidad_mostrar:.2f}", (50, 50), scale=3)
        
        # Dibujar el tablero
        for i in range(len(canasta_view_2D) - 1):
            cv.line(frame, tuple(canasta_view_2D[i]), tuple(canasta_view_2D[i + 1]), (0, 0, 255), 2)
            
        # Dibujar puntos de referencia
        for i, point in enumerate(referencias_imagen):
            cv.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        
        # Dibujar el aro
        for i in range(len(aro_2d) - 1):
            cv.line(frame, tuple(aro_2d[i]), tuple(aro_2d[i + 1]), (0, 0, 255), 2)
            
        # Dibujar el fondo y el cuadrado (pista)
        for i in range(len(fondo_2d) - 1):
            cv.line(frame, tuple(fondo_2d[i]), tuple(fondo_2d[i + 1]), (0, 255, 0), 2)
        
        for i in range(len(cuadrado_2d) - 1):
            cv.line(frame, tuple(cuadrado_2d[i]), tuple(cuadrado_2d[i + 1]), (0, 255, 0), 2)
        
        # ---------------------- BUCLE KALMAN -----------------------
        # Actualizar el filtro de Kalman si está activo
        if state['kalman_active']:
            # Actualizar el filtro con o sin medición
            measurement = np.array([center_x, center_y]) if center_x is not None else None
            mu, P, pred = ukf(mu, P, f, Q, b, a, measurement, h, R)
            print("------------------")
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
                
            # Verificar si la pelota ha bajado por debajo del aro y está descendiendo
            if (mu[2] < (centro_z + 0.1)) and (mu[5] < 0) and (frame_counter > 20):
                state['kalman_active'] = False  # Detener el seguimiento
                
            if calcular_probabilidad:
                num_parabolas = 100
                
                pasos_restantes = max(1, len(t) - frame_counter)  # Asegurarse de que no sea cero
                
                t_recortado = t[:pasos_restantes]  # Recortar el tiempo para las trayectorias
                
                # Simular posibles trayectorias
                x_sim, y_sim, z_sim = simular_trayectorias(mu, P, num_parabolas, t=t_recortado)
                
                # Verificar intersecciones con el aro
                trayectorias_intersectadas = verificar_interseccion(x_sim, y_sim, z_sim, aro, radio_aro)
                
                probabilidad = len(trayectorias_intersectadas) / num_parabolas
                
                # Actualizar el máximo si la nueva probabilidad es mayor
                if probabilidad > max_probabilidad:
                    max_probabilidad = probabilidad
                    
                probabilidad_mostrar = max_probabilidad
                
                frame_counter += 1
            
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
        plot3(ax2, fondo, 'g')
        plot3(ax2, cuadrado, 'g')
        plot3(ax2, camline, 'b')
        
        # Dibujar la trayectoria estimada si el filtro de Kalman está activo
        if impresion['print']:
            mu_filtered = np.array([mu[:3] for mu, _ in res])
            if len(mu_filtered) > 0:
                plot3(ax2, mu_filtered, 'g')
                
            for i in range(num_parabolas):
                color = 'cyan' if i in trayectorias_intersectadas else 'gray'
                plot3(ax2, np.vstack((x_sim[i], y_sim[i], z_sim[i])).T, color)
                
        set_axes_equal(ax2)
        
        # Mostrar la probabilidad en el frame
        putText(frame, f"Prob. canasta: {probabilidad_mostrar:.2f}", orig=(1000, 1000), scale=5)

        # Redibujar la figura y procesar eventos
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.001)

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