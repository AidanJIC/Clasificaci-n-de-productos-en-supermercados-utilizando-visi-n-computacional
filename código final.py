import cv2
import pyttsx3
from ultralytics import YOLO
import time

# Inicializar el motor de texto a voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad del habla
engine.setProperty('volume', 1.0)  # Volumen (0.0 a 1.0)

# Ruta al archivo best.pt
model_path = r'C:\Users\aidan\best.pt'

# Cargar el modelo YOLO
model = YOLO(model_path)

# Abrir la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

# Función para reproducir audio
def hablar(mensaje):
    engine.say(mensaje)
    engine.runAndWait()

# Inicializar temporizador para control de audio
tiempo_ultimo_audio = 0
intervalo_audio = 12  # Intervalo en segundos

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara.")
        break

    # Realizar la detección con el modelo YOLO
    results = model.predict(frame)
    
    # Obtener las clases detectadas
    detecciones = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []
    clases_detectadas = [model.names[int(det[5])] for det in detecciones] if len(detecciones) > 0 else []

    # Verificar si ha pasado el intervalo para reproducir audio
    tiempo_actual = time.time()
    if tiempo_actual - tiempo_ultimo_audio >= intervalo_audio:
        if clases_detectadas:
            mensaje = f"Objeto detectado: {', '.join(clases_detectadas)}"
            hablar(mensaje)
        else:
            hablar("No se detectó ningún objeto.")
        tiempo_ultimo_audio = tiempo_actual

    # Mostrar el frame con las detecciones
    for det in detecciones:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[int(cls)]} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Detección YOLO', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
