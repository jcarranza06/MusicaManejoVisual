import cv2
import numpy as np
#import tensorflow as tf
import pyttsx3

cap= cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_eye.xml')

umbral1 = 3
umbral2 = 3
a1=30
a2=255
mscale=20

testsSize = 50
indiceActual = 0
lados = [
    {"name":"arriba","samples":[]},
    {"name":"abajo","samples":[]},
    {"name":"izquierda","samples":[]},
    {"name":"derecha","samples":[]}
]
data_training = {"X":[],"y":[]}



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
        ojos = eye_cascade.detectMultiScale(roi_gray,1.3,12,maxSize=(mscale, mscale))
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
                #cv2.putText(roi_color, lados[indiceActual]["name"], (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.rectangle(roi_color, (x+x1,y+y1), (x+ancho+x1, y+alto+y1),(0,255,0),1)
                cv2.line(roi_color, (x1+x+int(ancho/2),0),(x1+x+int(ancho/2),al),(0,0,255),1)
                cv2.line(roi_color, (0, y1+y+int(ancho/2)),(an,y1+y+int(ancho/2)),(0,0,255),1)
                break
                #cv2.putText(roi_color, lados[np.argmax(p[0])]["name"], (x1, y1), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
               
                
            break
        break
    
    cv2. imshow('Deteccion de Rostro y ojos' ,frame)
    t=cv2.waitKey(1)
    if(t==27):
        break
    elif t == 119:  # Flecha arriba
        print("Flecha arriba presionada")
        print(mscale)
        mscale+=2
    elif t == 115:  # Flecha abajo
        print("Flecha abajo presionada")
        print(mscale)
        mscale-=2
    elif t == 97:  # Flecha izquierda
        print("Flecha izquierda presionada")
        a2-=1
    elif t == 100:  # Flecha derecha
        print("Flecha derecha presionada")
        a2+=1


cap.release()
cv2.destroyAllWindows()




