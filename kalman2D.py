import cv2
from ultralytics import YOLO
import numpy as np
from umucv.util import putText
from umucv.kalman import kalman

# Centro y semiejes de la elipse del aro
h, k = 1610, 428  # Centro del aro
c, d = 52, 10     # Semiejes de la elipse

# Definir los límites del aro en X (para filtrar raíces irrelevantes)
x_min, x_max = h - c, h + c

flag = False
calcular_probabilidad = True
# Variable para almacenar la probabilidad máxima alcanzada
max_probabilidad = 0
probabilidad_mostrar = 0

# Abrir el video
video_path = "canasta_2D_acierto.mp4"
cap = cv2.VideoCapture(video_path)

# Obtener información del video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# estado que Kalman va actualizando. Este es el valor inicial
degree = np.pi/180
a = np.array([0, 1650])

fps = int(cap.get(cv2.CAP_PROP_FPS))
dt = 1/fps
t = np.arange(0, 3 + dt,dt)
noise = 3

F = np.array(
    [1, 0,  dt,  0,
     0, 1,  0, dt,
     0, 0,  1,  0,
     0, 0,  0,  1 ]).reshape(4,4)

B = np.array(
         [dt**2/2, 0,
          0,       dt**2/2,
          dt,      0,
          0,       dt      ]).reshape(4,2)

H = np.array(
    [1,0,0,0,
     0,1,0,0]).reshape(2,4)

mu = np.array([0,0,0,0])  # Estado inicial

P  = np.diag([100,100,100,100])**2

res=[]
N = 15 # para tomar un tramo inicial y ver que pasa si luego se pierde la observacion

sigmaM = 0.01 # ruido del modelo
sigmaZ = 3*noise

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

listCenterX=[]
listCenterY=[]
listpuntos=[]

probabilidad = 0
frame_counter = 0

# Cargar el modelo entrenado
model = YOLO("runs/detect/train/weights/best.pt")

# Configurar el writer para guardar el video procesado
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Procesar el video frame por frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Terminar si no hay más frames

    # Realizar tracking en el frame
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="cuda", conf=0.6, iou=0.5, max_det=100, verbose=False)

    center_x = None
    center_y = None
    
    # Dibujar la elipse del aro
    cv2.ellipse(frame, (int(h), int(k)), (int(c), int(d)), 0, 0, 360, (255, 0, 0), 2)  # Azul

    # Obtener detecciones y modificar el frame
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
                
                # Calcular el centro de la detección
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Dibujar un círculo en el centro de la detección
                cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)
                
                
    if flag == True: 
        if center_x is not None and center_y is not None:  
            mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([center_x, center_y]),H,R) 
            m="normal"
            mm=True
        else :
            mu,P,pred= kalman(mu,P,F,Q,B,a,None,H,R)
            m="None"
            mm=False
            
        # Verificar si la pelota ha bajado por debajo del aro y está descendiendo
        if center_y > (k + d  - 100) and mu[3] > 0 and frame_counter > 20:  # mu[3] > 0 indica que la pelota está descendiendo
            calcular_probabilidad = False  
            
        if calcular_probabilidad:
                
            # Parámetros para simular las trayectorias
            num_parabolas = 100  # Número de trayectorias simuladas
            gravedad_px = 1650 # Gravedad en píxeles por segundo²
            
            # Generar trayectorias en un solo paso vectorizado
            x0 = np.random.normal(mu[0], np.sqrt(P[0, 0]), num_parabolas)  # Posiciones iniciales x
            y0 = np.random.normal(mu[1], np.sqrt(P[1, 1]), num_parabolas)  # Posiciones iniciales y
            vx0 = np.random.normal(mu[2], np.sqrt(P[2, 2]), num_parabolas)  # Velocidades iniciales x
            vy0 = np.random.normal(mu[3], np.sqrt(P[3, 3]), num_parabolas)  # Velocidades iniciales y
            
            # Simular trayectorias
            x_sim = x0[:, None] + vx0[:, None] * t  # Esto genera un array (num_parabolas, tiempo_sim)
            y_sim = y0[:, None] + vy0[:, None] * t + 0.5 * gravedad_px * t**2  # Efecto de la gravedad
            
            # Verificar intersección con la elipse (todo vectorizado)
            # Calculamos la expresión de la elipse para cada punto en las trayectorias
            distances = ((x_sim - h)**2 / c**2 + (y_sim - k)**2 / d**2)
            
            # La intersección ocurre cuando la distancia está menor o igual a 1
            intersects = np.any(distances <= 1, axis=1)  # Resultado para cada parábola (True si intersecta)

            # Calcular la probabilidad de intersección
            probabilidad = np.sum(intersects) / num_parabolas
        
            # Dibujar las trayectorias como líneas
            for i in range(num_parabolas):
                traj = np.vstack((x_sim[i], y_sim[i])).T
                traj = traj[(traj[:, 0] >= 0) & (traj[:, 0] < frame_width) & (traj[:, 1] >= 0) & (traj[:, 1] < frame_height)]  # Filtrar fuera de los límites

                if traj.shape[0] > 1:
                    cv2.polylines(frame, [np.int32(traj)], isClosed=False, color=(0, 255, 255), thickness=1)

                # Verificar intersección con la elipse del aro
                intersects = any(((x - h)**2 / c**2 + (y - k)**2 / d**2) <= 1 for x, y in traj)
                
            # Actualizar el máximo si la nueva probabilidad es mayor
            if probabilidad > max_probabilidad:
                max_probabilidad = probabilidad

            # Mostrar solo la probabilidad máxima alcanzada
            probabilidad_mostrar = max_probabilidad
                       
        if(mm):
            listCenterX.append(center_x)
            listCenterY.append(center_y)
            
        listpuntos.append((center_x,center_y,m))
        res += [(mu,P)]

        mu2 = mu
        P2 = P
        res2 = []
        
        predicciones_anteriores = None
        
        for _ in range(fps*2):
            mu2,P2,pred2= kalman(mu2,P2,F,Q,B,a,None,H,R)
            res2 += [(mu2,P2)]
               
        xe = [mu[0] for mu,_ in res]
        xu = [2*np.sqrt(P[0,0]) for _,P in res]
        ye = [mu[1] for  mu,_ in res]
        yu = [2*np.sqrt(P[1,1]) for _,P in res]   
        
        xp=[mu2[0] for mu2,_ in res2]
        yp=[mu2[1] for mu2,_ in res2]

        xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
        ypu = [2*np.sqrt(P[1,1]) for _,P in res2]
        
        for n in range(len(listCenterX)): # centro del roibox
            cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)
            
        for n in range(len(xe)): # xe e ye estimada
            incertidumbre=(xu[n]+yu[n])/2
            cv2.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),
            (255, 255, 0),1)
            
        for n in range(len(xp)): # x e y predicha
            incertidumbreP=(xpu[n]+ypu[n])

            cv2.circle(frame,(int(xp[n]),int(yp[n])),int(incertidumbreP),(0, 0, 255))
        
        frame_counter += 1    
        # Mostrar la probabilidad en el frame
    putText(frame, f"Probabilidad de interseccion: {probabilidad_mostrar:.2f}", (50, 50), scale= 2)
                     
    # Mostrar el frame modificado en tiempo real
    cv2.imshow("Tracking", frame)

    # Guardar el frame modificado
    out.write(frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("m"): 
        flag = True 

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
