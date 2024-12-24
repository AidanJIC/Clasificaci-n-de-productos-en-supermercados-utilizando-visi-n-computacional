from ultralytics import YOLO

if _name_ == "_main_":
    # Ruta al archivo de configuración de datos
    data_yaml_path = "C:/users/aidan/data.yaml"
    
    # Configuración del modelo YOLO
    model = YOLO("yolov8s.yaml")  # Puedes cambiar "yolov8n.yaml" por cualquier otro modelo YOLOv8 como "yolov8s.yaml", etc.

    # Entrenamiento del modelo
    results = model.train(
        data=data_yaml_path,  # Ruta al archivo data.yaml
        epochs=200,           # Número de épocas
        imgsz=640,            # Tamaño de las imágenes
        batch=16,             # Tamaño del batch
        name="custom_yolo_model",  # Nombre del proyecto o carpeta donde se guardarán los resultados
        workers=2             # Número de trabajadores para cargar datos (ajustar si se requiere)
    )

    # Mostrar resultados al final del entrenamiento
    print("Entrenamiento finalizado. Resultados guardados en la carpeta de ejecución del modelo.")