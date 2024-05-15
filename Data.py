import cv2
import os

# Importar la Clase (asegúrate de que SeguimientoManos.py esté en el mismo directorio)
import SeguimientoManos as sm


# Creación de la Carpeta
nombre = 'Letra_A'
direccion = 'C:/Users/lisan/OneDrive/Desktop/lenguajeSenas/Data'
carpeta = direccion + '/' + nombre

# Si no está creada la carpeta
if not os.path.exists(carpeta):
    print("CARPETA CREADA:", carpeta)
    # Crea la carpeta
    os.makedirs(carpeta)

# Lectura de la Cámara
cap = cv2.VideoCapture(0)
# Cambiamos la Resolución
cap.set(3, 1280)
cap.set(4, 720)

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

        cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),2)



    cv2.imshow("LENGUAJE DE SEÑAS", frame)
    # Leer teclado
    t = cv2.waitKey(1)
    if t == 27:  # Presiona la tecla 'Esc' para salir
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
