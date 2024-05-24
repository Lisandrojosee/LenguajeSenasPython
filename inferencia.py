import cv2
import os
from ultralytics import YOLO

# Importar la Clase (asegúrate de que SeguimientoManos.py esté en el mismo directorio)
import SeguimientoManos as sm

# Lectura de la Cámara
cap = cv2.VideoCapture(0)
# Cambiamos la Resolución
cap.set(3, 1280)
cap.set(4, 720)

#leer nuestro modelo
model = YOLO ('letras.pt')


# Declarar Detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    # Realizar la lectura de la captura
    ret, frame = cap.read()

    #Extraer informacion de la mano
    frame = detector.encontrarmanos(frame, dibujar= False)

    #posicion de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox= False, color=[0, 0,255])

    #si hay mano
    if mano == 1:
        #extraer informacion
        xmin, ymin, xmax, ymax = bbox

        #Asignamos Margen
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        #Relizar el recorte de nuestra mano
        recorte = frame[ymin:ymax, xmin:xmax]

        #Redimencionamiento
        recorte = cv2.resize(recorte, (640, 640), interpolation = cv2.INTER_CUBIC)

        #Extraer resultados
        resultados = model.predict(recorte, conf=0.55)
        if len(resultados) !=0:
            #iteramos
            for results in resultados:
                masks = results.masks
            coordenadas = masks

            anotaciones = resultados[0].plot()

            cv2.imshow('RECORTE', anotaciones)





    cv2.imshow("LENGUAJE DE SEÑAS", frame)
    # Leer teclado
    t = cv2.waitKey(1)
    if t == 27:  # Presiona la tecla 'Esc' para salir
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()