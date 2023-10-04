import cv2
import os
import imutils
import shutil  # Importar la biblioteca shutil para eliminar la carpeta

# Ruta donde se guarda
dataPath = 'D:/OneDrive - AXEDE/capturas'

# Función para manejar la entrada de texto
def get_name_from_user():
    name = input("Ingrese su nombre y presione Enter (o 'q' para salir): ")
    return name

# Capturar el nombre desde el usuario
name = get_name_from_user()

if name.lower() == 'q':
    cv2.destroyAllWindows()
else:
    # Ruta donde se guarda
    personPath = os.path.join(dataPath, name)

    # Verificar si la carpeta ya existe y eliminarla si es necesario
    if os.path.exists(personPath):
        shutil.rmtree(personPath)

    # Crear una nueva carpeta
    os.makedirs(personPath)
    print("Carpeta creada:", personPath)

    # Iniciar la captura de video desde la cámara 0, 1 y 2 (cambia los números según las cámaras que desees usar)
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)

    # Clasificador para detección de rostros
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret0 and ret1 and ret2):
            break

        # Ajustar el tamaño de los frames para visualización
        frame0 = imutils.resize(frame0, width=720)
        frame1 = imutils.resize(frame1, width=720)
        frame2 = imutils.resize(frame2, width=720)

        # Convertir a escala de grises
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Crear copias de los frames para visualización en tiempo real
        frame_display0 = frame0.copy()
        frame_display1 = frame1.copy()
        frame_display2 = frame2.copy()

        # Detectar rostros en cada cámara
        faces_detected_cam0 = len(faceClassif.detectMultiScale(gray0, 1.3, 5)) > 0
        faces_detected_cam1 = len(faceClassif.detectMultiScale(gray1, 1.3, 5)) > 0
        faces_detected_cam2 = len(faceClassif.detectMultiScale(gray2, 1.3, 5)) > 0

        # Si se detecta un rostro en la cámara 0, guardar imágenes de las tres cámaras
        if faces_detected_cam0:
            for (x, y, w, h) in faceClassif.detectMultiScale(gray0, 1.3, 5):
                # Aumenta la región de recorte para incluir más del cuello, pelo y hombros
                y_start = max(0, y - 50)
                y_end = min(frame0.shape[0], y + h + 50)
                x_start = max(0, x - 50)
                x_end = min(frame0.shape[1], x + w + 50)

                body0 = frame0[y_start:y_end, x_start:x_end]
                body0 = cv2.resize(body0, (2048, 2048), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(personPath, 'cam0_{}_{}.jpg'.format(name, count)), body0)

                # Tomar la toma completa de la cámara 1
                cv2.imwrite(os.path.join(personPath, 'cam1_{}_{}.jpg'.format(name, count)), frame1)

                # Tomar la toma completa de la cámara 2
                cv2.imwrite(os.path.join(personPath, 'cam2_{}_{}.jpg'.format(name, count)), frame2)

                count += 1

        # Dibujar el cuadro verde en los frames de visualización si se detectan rostros
        if faces_detected_cam0:
            for (x, y, w, h) in faceClassif.detectMultiScale(gray0, 1.3, 5):
                cv2.rectangle(frame_display0, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if faces_detected_cam1:
            for (x, y, w, h) in faceClassif.detectMultiScale(gray1, 1.3, 5):
                cv2.rectangle(frame_display1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if faces_detected_cam2:
            for (x, y, w, h) in faceClassif.detectMultiScale(gray2, 1.3, 5):
                cv2.rectangle(frame_display2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar los frames en tiempo real
        cv2.imshow('Cámara 0', frame_display0)
        cv2.imshow('Cámara 1', frame_display1)
        cv2.imshow('Cámara 2', frame_display2)

        k = cv2.waitKey(1)
        if k == ord('q') or count >= 50:  # 'q' para salir o límite de 150 imágenes (50 por cualquier cámara)
            break

    # Liberar las cámaras y cerrar las ventanas
    cap0.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
