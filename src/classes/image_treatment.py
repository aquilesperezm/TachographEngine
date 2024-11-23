import cv2
import numpy as np
import pytesseract
from skimage.feature import canny


class ImageTreatment:
    def __init__(self, image):

        self.min_threshold = None
        self.max_threshold = None
        self.edges = None
        self.image = image
        self.contours = None

        # detection properties
        self.use_canny = False
        self.noise_reduction = False

        self.apply_smoothing = False
        self.smoothing_coefficient = -3

    def get_image(self):
        return self.image

    def set_detection_properties(self, use_canny: bool = False, noise_reduction: bool = False,
                                 apply_smoothing: bool = False):
        self.use_canny = use_canny
        self.noise_reduction = noise_reduction
        self.apply_smoothing = apply_smoothing

    """
    the smoothing coefficient must be negative 
    """
    def set_smoothing_coefficient(self,coefficient: int):
        self.smoothing_coefficient = coefficient


    def extract_center(self, square_size: int, x_1: int = 0, y_1: int = 0, x_2: int = 0, y_2: int = 0):

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

    def __generate_edges(self):

        # with canny detector
        if self.use_canny:
            if self.noise_reduction:
                img_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
                self.edges = cv2.Canny(img_blur, self.min_threshold, self.max_threshold)
            else:
                self.edges = cv2.Canny(self.image, self.min_threshold, self.max_threshold)

        # with threshold detector
        else:
            if self.noise_reduction:
                img_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
                _, thresh = cv2.threshold(img_blur, self.min_threshold, self.max_threshold, cv2.THRESH_BINARY_INV)
                self.edges = thresh
            else:
                _, thresh = cv2.threshold(self.image, self.min_threshold, self.max_threshold, cv2.THRESH_BINARY_INV)
                self.edges = thresh


    def generate_contours(self):

        # ----------------------------------- Reduccion de Ruido ---------------------------------------------------
        self.__generate_edges()

        # Encontrar contornos
        contours, hierarchy = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --------------------------------- Suavizado de Contornos------------------------------------------------------
        result = []
        if self.apply_smoothing:
            # Aproximación de contornos usando Ramer-Douglas-Peucker

            for cnt in contours:
                epsilon = (10 ** self.smoothing_coefficient) * cv2.arcLength(cnt,
                                                      True)  # Ajusta el valor de epsilon para controlar la simplificación
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                result.append(approx)
            # --------------------------------------------------------------------------------------------------------------

        if self.apply_smoothing:
            self.contours = result
        else:
            self.contours = result

        return self.contours, self.edges

    def draw_contours(self, title: str, contours, thickness: int, color) -> None:
        cv2.drawContours(self.image, contours, -1, color, thickness)
        cv2.imshow(title, self.image)
        cv2.waitKey(0)

    def show_details_contour(self, contour):
        # Calcular el área
        # area = cv2.contourArea(contour)
        # print(f'Área del contorno: {area} píxeles')

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
        # contours, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    cv2.putText(self.image, f"Angulo: {angulo:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                1)
                    print(f"Angulo: {angulo:.2f}" + " x: " + str(x) + " y: " + str(y))

                # Verificar intersección con el eje Y
                if x == ancho // 2:
                    cv2.circle(self.image, (x, y), 5, (0, 255, 255), -1)  # Marcar intersección en rojo
                    cv2.putText(self.image, f"Angulo: {angulo:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                1)
                    print(f"Angulo: {angulo:.2f}" + " x: " + str(x) + " y: " + str(y))

            # Dibujar el contorno
            cv2.drawContours(self.image, contours, i, (0, 255, 0), 2)

        # Mostrar la imagen
        cv2.imshow('Contornos y Ángulos', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotate_image(self, grados):
        # Cargar la imagen
        # imagen = cv2.imread('ruta/a/tu/imagen.jpg')

        # Obtener las dimensiones de la imagen
        (h, w) = self.image.shape[:2]

        # Definir el centro de rotación
        centro = (w // 2, h // 2)

        # Definir el ángulo de rotación
        # grados = 45  # Cambia este valor a los grados que desees
        angulo = -grados  # Negativo para rotar en sentido horario

        # Obtener la matriz de rotación
        matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

        # Rotar la imagen
        self.image = cv2.warpAffine(self.image, matriz_rotacion, (w, h))

        # Guardar la imagen rotada
        # cv2.imwrite('ruta/a/tu/imagen_rotada.jpg', imagen_rotada)

        # Mostrar la imagen rotada (opcional)
        cv2.imshow('Imagen Rotada', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_contours(self):
        return self.contours

    """
    oem - OCR Engine Mode
      0 = Original Tesseract only.
      1 = Neural nets LSTM only.
      2 = Tesseract + LSTM.
      3 = Default, based on what is available.
  psm - Page Segmentation Mode
      0 = Orientation and script detection (OSD) only.
      1 = Automatic page segmentation with OSD.
      2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
      3 = Fully automatic page segmentation, but no OSD. (Default)
      4 = Assume a single column of text of variable sizes.
      5 = Assume a single uniform block of vertically aligned text.
      6 = Assume a single uniform block of text.
      7 = Treat the image as a single text line.
      8 = Treat the image as a single word.
      9 = Treat the image as a single word in a circle.
      10 = Treat the image as a single character.
      11 = Sparse text. Find as much text as possible in no particular order.
      12 = Sparse text with OSD.
      13 = Raw line. Treat the image as a single text line,
          bypassing hacks that are Tesseract-specific.
      
      example:
          tess_string = pytesseract.image_to_string(img, config=f'--oem {oem} --psm {psm}')
          regex_result = re.findall(r'[A-Z0-9]', tess_string) # filter only uppercase alphanumeric symbols
          return ''.join(regex_result)    
          
    """
    def detect_number(self,target_number:str):

        if self.use_canny:
            raise Exception('This method not use canny detection, please deactivate canny detection')

        if self.contours is None:
            raise Exception('You must generate contours and edges, execute generate_contours first')


        # Variable para almacenar el resultado
        detected_number = ""

        # Dibujar contornos y detectar el número "12"
        print("Starting analysis: ...")

        i = 0
        for contour in self.contours:


            # Obtener el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)

            # Extraer la región de interés (area del contorno)
            roi = self.edges[y:y + h, x:x + w]
            # roi = cv2.contourArea(contour)

            # Usar Tesseract para reconocer texto en la región de interés
            text = pytesseract.image_to_string(roi, config='--psm 6 outputbase digits')
            detected_number = text.strip()

            i = i + 1
            print(str(i) + " / " + str(len(self.contours)) + " : " + detected_number)

            #if detected_number != "":
            #print("Numero detectado: " + str(detected_number))

            # Verificar si el texto reconocido es "12"
            if text.strip() == target_number:
                detected_number = text.strip()
                # Dibujar el rectángulo alrededor del número detectado
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar el resultado
        cv2.imshow('Detección de contornos y reconocimiento', self.image)
        # print(f"Número detectado: {detected_number}")
        print("Done!!!!")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def set_threshold(self, min_threshold: int, max_threshold: int):
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

    def get_threshold(self):
        return self.min_threshold, self.max_threshold
