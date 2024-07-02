import cv2
import numpy as np
import pandas as pd
# import tensorflow as tf
import pyttsx3
import socket

# Configuración del cliente
host = 'localhost'
port = 30000

# Crea un socket TCP/IP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Conéctate al servidor
client_socket.connect((host, port))

cap= cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_eye.xml')

umbral1 = 3
umbral2 = 3
a1=50
a2=255

def clusterData(data, n_clusters):
  # funcion lambda para distancia pitagorica entre dos puntos en r_n
  distancia = lambda arr1, arr2: np.sqrt(np.sum((arr1 - arr2)**2))

  #se inicializan los clusters con los primerso elementos del arreglo
  centroids = data[:n_clusters].astype(float)
  
  while True:
    #se le asigna un cluster a cada elemento de data
    x = np.array([np.argmin([distancia(i,y) for y in centroids]) for i in data]) # por defecto se elige el centroide con met¿nor indice

    # para cada cluster se saca la media, la posiciion del centriode
    newValues = np.array([np.mean(data[x == i], axis=0) for i in range(len(centroids))])

    #en caso de que no cmbien los centroides se devuelven los valores acctuales
    if np.array_equal(centroids, newValues):
        print("Se alcanzó el valor límite. Terminando el bucle.")
        return(centroids, x)
    #se actualizan los centroides
    mask = ~np.isnan(newValues) # punto b solo cambia para valores no nulos
    centroids[mask] = newValues[mask]

  #return generico
  return(centroids, data)

def clasifyToCentroid(centroids, data):
  distancia = lambda arr1, arr2: np.sqrt(np.sum((arr1 - arr2)**2))
  return np.array([np.argmin([distancia(i,y) for y in centroids]) for i in data])


# Inicializar el motor de texto a voz
engine = pyttsx3.init()

# Configurar propiedades del habla
engine.setProperty('rate', 150)  # Velocidad de habla (palabras por minuto)
engine.setProperty('volume', 1)  # Volumen (0.0 a 1.0)

# Función para hablar un texto
def hablar(texto):
    engine.say(texto)
    engine.runAndWait()

testsSize = 60
indiceActual = 0
lados = [
    {"name":"do","samples":[]},
    {"name":"re","samples":[]},
    {"name":"mi","samples":[]},
    {"name":"fa","samples":[]},
    {"name":"sol","samples":[]},
    {"name":"la","samples":[]},
    {"name":"si","samples":[]}
]
data_training = {"X":[],"y":[]}
hablar("acomodese y calibre el brillo con w y s, termine con esc")
while True:
    ret, frame = cap.read()
    added = False
    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    caras = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in caras:
        #Dibujando los rectangulos de detecciona de la cara
        cv2.putText(frame, 'Rostro' ,(x,y-20),2,0.5, (255,0,0),1, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame [y:y+h, x:x+w]
        ojos = eye_cascade.detectMultiScale(roi_gray,1.3,8,maxSize=(64, 64))
        for (ох, oy, ow, oh) in ojos:

            x1 = ох
            x2 = ох+ow
            y1 = oy
            y2 = oy+oh
            al,an,c = roi_color.shape
            #cv2.putText(roi_color, "rec", (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.rectangle(roi_color, (x1,y1),(x2,y2),(0,255,0),2)
            recorte = roi_color[y1:y2, x1:x2]
            gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            gris = cv2.GaussianBlur(gris, (umbral1,umbral2), 0)
            _, umbral =cv2.threshold(gris,a1,a2,cv2.THRESH_BINARY_INV)
            contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

            for contorno in contornos:
                (x,y,ancho,alto)=cv2.boundingRect(contorno)
                #print(x,y, ancho, alto)www
                if(ancho > 20 or alto > 20 or ancho/alto > 1.69):
                    break
                if(indiceActual < len(lados)):
                    cv2.putText(roi_color, lados[indiceActual]["name"], (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.rectangle(roi_color, (x+x1,y+y1), (x+ancho+x1, y+alto+y1),(0,255,0),1)
                cv2.line(roi_color, (x1+x+int(ancho/2),0),(x1+x+int(ancho/2),al),(0,0,255),1)
                cv2.line(roi_color, (0, y1+y+int(ancho/2)),(an,y1+y+int(ancho/2)),(0,0,255),1)
                
                break
            break
        break
    
    if(added):
        print(lados)
    cv2. imshow('Deteccion de Rostro y ojos' ,frame)
    t=cv2.waitKey(1)
    if(t==27):
        break
    elif t == 119:  # Flecha arriba
        print("Flecha arriba presionada")
        print(a1)
        a1+=1
    elif t == 115:  # Flecha abajo
        print("Flecha abajo presionada")
        print(a1)
        a1-=1
    elif t == 97:  # Flecha izquierda
        print("Flecha izquierda presionada")
        a2-=1
    elif t == 100:  # Flecha derecha
        print("Flecha derecha presionada")
        a2+=1

hablar("vamos a entrenar el modelo, mire hacia do")
while True:
    ret, frame = cap.read()
    added = False
    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    caras = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in caras:
        #Dibujando los rectangulos de detecciona de la cara
        cv2.putText(frame, 'Rostro' ,(x,y-20),2,0.5, (255,0,0),1, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame [y:y+h, x:x+w]
        ojos = eye_cascade.detectMultiScale(roi_gray,1.3,8,maxSize=(64, 64))
        for (ох, oy, ow, oh) in ojos:
        #Dibujando los rectangulos de detecciona de los ojos
            #cv2.putText(frame, 'Ojos', (x,y+60) ,2,0.5, (0,255,0),1,cv2.LINE_AA)
            #cv2.rectangle(roi_color,(ох, oy),(ох+ow,oy+oh), (0,255,0),5)

            x1 = ох
            x2 = ох+ow
            y1 = oy
            y2 = oy+oh
            al,an,c = roi_color.shape
            #cv2.putText(roi_color, "rec", (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.rectangle(roi_color, (x1,y1),(x2,y2),(0,255,0),2)
            recorte = roi_color[y1:y2, x1:x2]
            gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
            gris = cv2.GaussianBlur(gris, (umbral1,umbral2), 0)
            _, umbral =cv2.threshold(gris,a1,a2,cv2.THRESH_BINARY_INV)
            contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

            for contorno in contornos:
                (x,y,ancho,alto)=cv2.boundingRect(contorno)
                #print(x,y, ancho, alto)
                if(ancho > 20 or alto > 20 or ancho/alto > 1.69):
                    break
                if(indiceActual < len(lados)):
                    cv2.putText(roi_color, lados[indiceActual]["name"], (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.rectangle(roi_color, (x+x1,y+y1), (x+ancho+x1, y+alto+y1),(0,255,0),1)
                cv2.line(roi_color, (x1+x+int(ancho/2),0),(x1+x+int(ancho/2),al),(0,0,255),1)
                cv2.line(roi_color, (0, y1+y+int(ancho/2)),(an,y1+y+int(ancho/2)),(0,0,255),1)
                
                if(indiceActual < len(lados) and len(lados[indiceActual]["samples"]) < testsSize):
                    added=True
                    #print(indiceActual)
                    #print(len(lados[indiceActual]["samples"]))
                    lados[indiceActual]["samples"].append([x, y])
                    data_training["X"].append([x, y])
                    data_training["y"].append(indiceActual)
                else:
                    if(indiceActual < len(lados)):
                        indiceActual+=1
                        if(indiceActual < len(lados)):
                            hablar("presione tecla, mire hacia "+lados[indiceActual]["name"])
                            input("Presiona Enter para continuar...")
                break
            break
        break
    
    if(indiceActual >= len(lados)):
        break
    if(added):
        print(lados)
    cv2. imshow('Deteccion de Rostro y ojos' ,frame)
    t=cv2.waitKey(1)
    if(t==27):
        break
    elif t == 119:  # Flecha arriba
        print("Flecha arriba presionada")
        print(a1)
        a1+=1
    elif t == 115:  # Flecha abajo
        print("Flecha abajo presionada")
        print(a1)
        a1-=1
    elif t == 97:  # Flecha izquierda
        print("Flecha izquierda presionada")
        a2-=1
    elif t == 100:  # Flecha derecha
        print("Flecha derecha presionada")
        a2+=1

print(data_training)

hablar("entrenando")

X = np.array(data_training["X"])
#y = tf.keras.utils.to_categorical(np.array(data_training["y"]), num_classes=4)
y = np.array(data_training["y"])
#model = tf.keras.Sequential(
#    [
#        tf.keras.layers.Dense(4, activation="tanh", input_shape=(2,)),
#        tf.keras.layers.Dense(4, activation="tanh"),
#        tf.keras.layers.Dense(4, activation="tanh"),
#        tf.keras.layers.Dense(4, activation="tanh")
#    ]
#)
#
#model.compile(
#    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
#    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#    metrics=["accuracy"],
#)
#
#model.fit(X, y, epochs=90, batch_size=1)
print(X)
print(y)
centroidesTemp = np.array([np.mean(X[y == i], axis=0) for i in range(len(lados))])
print(centroidesTemp)
print(np.concatenate((centroidesTemp, X), axis=0))
centroides,elems = clusterData(np.concatenate((centroidesTemp, X), axis=0), 7)

predicciones=[]

try:
    while True:
        ret, frame = cap.read()
        added = False
        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in caras:
            #Dibujando los rectangulos de detecciona de la cara
            cv2.putText(frame, 'Rostro' ,(x,y-20),2,0.5, (255,0,0),1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),5)
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame [y:y+h, x:x+w]
            ojos = eye_cascade.detectMultiScale(roi_gray,1.3,8,maxSize=(64, 64))
            for (ох, oy, ow, oh) in ojos:
            #Dibujando los rectangulos de detecciona de los ojos
                #cv2.putText(frame, 'Ojos', (x,y+60) ,2,0.5, (0,255,0),1,cv2.LINE_AA)
                #cv2.rectangle(roi_color,(ох, oy),(ох+ow,oy+oh), (0,255,0),5)

                x1 = ох
                x2 = ох+ow
                y1 = oy
                y2 = oy+oh
                al,an,c = roi_color.shape
                #cv2.putText(roi_color, "rec", (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.rectangle(roi_color, (x1,y1),(x2,y2),(0,255,0),2)
                recorte = roi_color[y1:y2, x1:x2]
                gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                gris = cv2.GaussianBlur(gris, (umbral1,umbral2), 0)
                _, umbral =cv2.threshold(gris,a1,a2,cv2.THRESH_BINARY_INV)
                contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

                for contorno in contornos:
                    (x,y,ancho,alto)=cv2.boundingRect(contorno)
                    if(ancho > 20 or alto > 20 or ancho/alto > 1.69):
                        break
                    #cv2.putText(roi_color, lados[indiceActual]["name"], (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.rectangle(roi_color, (x+x1,y+y1), (x+ancho+x1, y+alto+y1),(0,255,0),1)
                    cv2.line(roi_color, (x1+x+int(ancho/2),0),(x1+x+int(ancho/2),al),(0,0,255),1)
                    cv2.line(roi_color, (0, y1+y+int(ancho/2)),(an,y1+y+int(ancho/2)),(0,0,255),1)
                    predicciones.append([x, y])
                    if(len(predicciones)> 7):
                        print(predicciones)
                        p = pd.Series(clasifyToCentroid(centroides, np.array(predicciones))).mode()[0]
                        cv2.putText(roi_color, lados[p]["name"], (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        hablar(lados[p]["name"])
                        print('++++++++++++++++++++++')
                        print(lados[p]["name"])
                        print('++++++++++++++++++++++')
                        client_socket.sendall(str(lados[p]["name"]).encode('utf-8'))

                        predicciones=[]
                    
                    break
                break
            break
        
        cv2. imshow('Deteccion de Rostro y ojos' ,frame)
        t=cv2.waitKey(1)
        if(t==27):
            break
        elif t == 119:  # Flecha arriba
            print("Flecha arriba presionada")
            print(a1)
            a1+=1
        elif t == 115:  # Flecha abajo
            print("Flecha abajo presionada")
            print(a1)
            a1-=1
        elif t == 97:  # Flecha izquierda
            print("Flecha izquierda presionada")
            a2-=1
        elif t == 100:  # Flecha derecha
            print("Flecha derecha presionada")
            a2+=1

finally:
    # Cierra la conexión
    client_socket.close()

cap.release()
cv2.destroyAllWindows()




