import cv2
import numpy as np
import pytesseract
from skimage.feature import canny


class ImageTreatment:
    def __init__(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def extract_center(self,square_size:int,x_1:int = 0,y_1:int = 0,x_2:int = 0,y_2:int = 0):

        # Obtener las dimensiones de la imagen
        alto, ancho = self.image.shape[:2]

        # Definir el tamaño del cuadro que deseas recortar (por ejemplo, 200x200)
        tamano_cuadro = square_size

        # Calcular las coordenadas del recorte horizontal
        x1 = (ancho - tamano_cuadro) // 2 + x_1
        y1 = (alto - tamano_cuadro) // 2 + y_1
        x2 = x1 + tamano_cuadro - x_2
        y2 = y1 + tamano_cuadro - y_2

        # Realizar el recorte
        imagen_recortada = self.image[y1:y2, x1:x2]

        self.image = imagen_recortada

    def get_edges(self, canny_threshold_min:int,canny_threshold_max:int,noise_reduction: bool):

        edges = None

        if noise_reduction:
            img_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
            edges = cv2.Canny(img_blur, canny_threshold_min, canny_threshold_max)
        # -----------------------------------------------------------------------------------------------------------
        else:
            # Detectar bordes con Canny
            edges = cv2.Canny(self.image, canny_threshold_min, canny_threshold_max)

        return edges

    def get_thresh(self,threshold_min:int,threshold_max:int,):
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar un umbral
        _, thresh = cv2.threshold(gray, threshold_min, threshold_max, cv2.THRESH_BINARY_INV)

        return thresh

    def detect_contours(self, canny_threshold_min:int,canny_threshold_max:int,use_canny:bool = False, order_area: bool = False, noise_reduction: bool = False, border_smoothed: bool = False):

        # Cargar la imagen
        # img_blur = cv2.imread(imagePath, 0)  # Cargar en escala de grises

        # ----------------------------------- Reduccion de Ruido ---------------------------------------------------



        if not use_canny and noise_reduction:
            img_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
            _, edges = cv2.threshold(img_blur, canny_threshold_min, canny_threshold_max, cv2.THRESH_BINARY_INV)
        elif not use_canny:
            _, edges = cv2.threshold(self.image, canny_threshold_min, canny_threshold_max, cv2.THRESH_BINARY_INV)
        else:
            edges = self.get_edges(canny_threshold_min, canny_threshold_max, noise_reduction)

        # Encontrar contornos
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --------------------------------- Suavizado de Contornos------------------------------------------------------
        # Aproximación de contornos usando Ramer-Douglas-Peucker
        result = []

        for cnt in contours:
            epsilon = 0.0000000000001 * cv2.arcLength(cnt, True)  # Ajusta el valor de epsilon para controlar la simplificación
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            result.append(approx)
        #--------------------------------------------------------------------------------------------------------------

        if border_smoothed:
           if order_area:
                return sorted(result, key=cv2.contourArea, reverse=True), edges
           else:
                return result, edges
        else:
            if order_area:
                return sorted(result, key=cv2.contourArea, reverse=True), edges

            else:
                return contours, edges

    def draw_contours(self, title: str,contours, thickness: int) -> None:
        cv2.drawContours(self.image, contours, -1, (0, 0, 255), thickness)
        cv2.imshow(title, self.image)
        cv2.waitKey(0)

    def show_details_contour(self, contour):
        # Calcular el área
        #area = cv2.contourArea(contour)
        #print(f'Área del contorno: {area} píxeles')

        # Encontrar el centroide del contorno
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Para evitar división por cero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Dibujar el contorno y su centroide
            cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)  # Contorno en verde
            cv2.circle(self.image, (cx, cy), 5, (255, 0, 0), -1)  # Centroide en azul

            # Trazar el eje central (horizontal y vertical)
            cv2.line(self.image, (0, cy), (self.image.shape[1], cy), (255, 0, 255), 2)  # Línea horizontal
            cv2.line(self.image, (cx, 0), (cx, self.image.shape[0]), (255, 0, 255), 2)  # Línea vertical

            # Mostrar grados alrededor del centroide
            for angulo in range(0, 360, 30):  # Cada 30 grados
                x = int(cx + 100 * np.cos(np.radians(angulo)))  # Radio de la línea
                y = int(cy + 100 * np.sin(np.radians(angulo)))
                cv2.putText(self.image, f'{angulo}°', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Mostrar la imagen con el eje central y los grados

        cv2.imshow('Eje Central y Grados', self.image)
        cv2.waitKey(0)

    def show_details_multiple_contours(self, contours):

        # Convertir a escala de grises

        # Encontrar contornos
        #contours, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar el eje de coordenadas
        alto, ancho = self.image.shape[:2]
        cv2.line(self.image, (0, alto // 2), (ancho, alto // 2), (255, 0, 0), 2)  # Eje X
        cv2.line(self.image, (ancho // 2, 0), (ancho // 2, alto), (255, 0, 0), 2)  # Eje Y

        # Procesar cada contorno
        for i, contorno in enumerate(contours):
            # Calcular el rectángulo mínimo que encierra el contorno
            rect = cv2.minAreaRect(contorno)
            angulo = rect[2]  # El ángulo del rectángulo

            # Ajustar el ángulo
            if angulo < -45:
                angulo += 90

            # Comprobar intersecciones con los ejes
            for punto in contorno:
                x, y = punto[0][0], punto[0][1]
                # Verificar intersección con el eje X
                if y == alto // 2:
                    cv2.circle(self.image, (x, y), 5, (0, 255, 255), -1)  # Marcar intersección en rojo
                    cv2.putText(self.image, f"Angulo: {angulo:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1)
                    print(f"Angulo: {angulo:.2f}" + " x: " + str(x) + " y: " + str(y))

                # Verificar intersección con el eje Y
                if x == ancho // 2:
                    cv2.circle(self.image, (x, y), 5, (0, 255, 255), -1)  # Marcar intersección en rojo
                    cv2.putText(self.image, f"Angulo: {angulo:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1)
                    print(f"Angulo: {angulo:.2f}" + " x: "+ str(x) + " y: "+ str(y))

            # Dibujar el contorno
            cv2.drawContours(self.image, contours, i, (0, 255, 0), 2)

        # Mostrar la imagen
        cv2.imshow('Contornos y Ángulos', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotate_image(self, grados):
        # Cargar la imagen
        #imagen = cv2.imread('ruta/a/tu/imagen.jpg')

        # Obtener las dimensiones de la imagen
        (h, w) = self.image.shape[:2]

        # Definir el centro de rotación
        centro = (w // 2, h // 2)

        # Definir el ángulo de rotación
        #grados = 45  # Cambia este valor a los grados que desees
        angulo = -grados  # Negativo para rotar en sentido horario

        # Obtener la matriz de rotación
        matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

        # Rotar la imagen
        self.image = cv2.warpAffine(self.image, matriz_rotacion, (w, h))

        # Guardar la imagen rotada
        #cv2.imwrite('ruta/a/tu/imagen_rotada.jpg', imagen_rotada)

        # Mostrar la imagen rotada (opcional)
        cv2.imshow('Imagen Rotada', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def detect_twelve(self,contours,enable_edges:bool = False):
        """
        test

        """
        # Configura la ruta de Tesseract si es necesario
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

        # Cargar la imagen
        #image = cv2.imread('ruta/a/tu/imagen.jpg')

        # Convertir a escala de grises
        #gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Aplicar un umbral
        #_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)


        #thresh = self.get_thresh(150,255)
        #if enable_edges:
        #    thresh = self.get_edges(150,255,True)

        # Encontrar contornos

        #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours, thresh = self.detect_contours(110,255,False,True,True,True)

        # Variable para almacenar el resultado
        detected_number = ""

        # Dibujar contornos y detectar el número "12"
        for contour in contours:
            # Obtener el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)

            # Extraer la región de interés (area del contorno)
            roi = thresh[y:y + h, x:x + w]
            #roi = cv2.contourArea(contour)

            # Usar Tesseract para reconocer texto en la región de interés
            text = pytesseract.image_to_string(roi, config='--psm 6 outputbase digits')
            detected_number = text.strip()
            if detected_number != "":
                print("Numero detectado: " + str(detected_number))

            # Verificar si el texto reconocido es "12"
            if text.strip() == "12":
                detected_number = text.strip()
                # Dibujar el rectángulo alrededor del número detectado
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar el resultado
        cv2.imshow('Detección de contornos y reconocimiento', self.image)
        #print(f"Número detectado: {detected_number}")
        print("Done!!!!")
        cv2.waitKey(0)
        cv2.destroyAllWindows()