import cv2
import os

# Importar la Clase (asegúrate de que SeguimientoManos.py esté en el mismo directorio)
import SeguimientoManos as sm


# Creación de la Carpeta
nombre = 'Letra_A'
direccion = 'C:/Users/lisan/Desktop/lenguajeSenas/Data'
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

#Declaramos Contador
count = 0

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
       #recorte = cv2.resize(recorte, (640, 640), interpolation = cv2.INTER_CUBIC)

        #Almacenar Nuestras Imagenes
        cv2.imwrite(carpeta + "/A_{}.jpg" .format(count), recorte)

        #Aumentamos Contador
        count = count + 1

        cv2.imshow("RECORTE", recorte)




    cv2.imshow("LENGUAJE DE SEÑAS", frame)
    # Leer teclado
    t = cv2.waitKey(1)
    if t == 27 or count == 100:  # Presiona la tecla 'Esc' para salir
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
