import cv2
import numpy as np
from matplotlib.pyplot import contour
from numpy import ndarray
import imutils
from src.classes.image_treatment import ImageTreatment

image_source = 'src\\assets\\images\\image1.jpg'
image = cv2.imread(image_source,0)

image_template = 'src\\assets\\images\\tear.jpg'
image2 = cv2.imread(image_template,0)

it = ImageTreatment(image)
#it.extract_center(330, 80, 70, 90, 150)
c,e = it.detect_contours(110, 255, False, True, True,True)
#print(len(c))
it.draw_contours("Test", c, 2)
#print(len(c))
#it.show_details_multiple_contours(c)
#it.detect_twelve(c)

"""           
def temp():

    # Cargar la imagen principal y la plantilla
    imagen_principal = cv2.imread(image_source)
    plantilla = cv2.imread(image_template)

    # Obtener las dimensiones de la plantilla
    alto_plantilla, ancho_plantilla = plantilla.shape[:2]

    # Realizar el template matching
    resultado = cv2.matchTemplate(imagen_principal, plantilla, cv2.TM_CCOEFF_NORMED)

    print(resultado)

    # Establecer un umbral para considerar una coincidencia
    umbral = 0.8
    ubicaciones = np.where(resultado >= umbral)

    # Dibujar un rectángulo alrededor de las coincidencias encontradas
    for pt in zip(*ubicaciones[::-1]):  # Cambiar ubicaciones a (x, y)
        cv2.rectangle(imagen_principal, pt, (pt[0] + ancho_plantilla, pt[1] + alto_plantilla), (0, 255, 0), 2)

        # Mostrar la imagen con las coincidencias
    cv2.imshow('Coincidencias', imagen_principal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def temp2():

    # Cargar la imagen principal y la plantilla
    imagen_principal = cv2.imread(image_source)
    plantilla = cv2.imread(image_template)

    # Tamaños a probar (puedes ajustar estos valores)
    niveles_de_escala = [0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5]

    # Lista para almacenar las ubicaciones de las coincidencias
    ubicaciones_coincidencias = []

    for escala in niveles_de_escala:
        # Redimensionar la plantilla
        alto_nuevo = int(plantilla.shape[0] * escala)
        ancho_nuevo = int(plantilla.shape[1] * escala)
        plantilla_redimensionada = cv2.resize(plantilla, (ancho_nuevo, alto_nuevo))

        # Realizar el template matching
        resultado = cv2.matchTemplate(imagen_principal, plantilla_redimensionada, cv2.TM_CCOEFF_NORMED)

        # Establecer un umbral para considerar una coincidencia
        umbral = 0.8
        ubicaciones = np.where(resultado >= umbral)

        # Añadir las ubicaciones a la lista y dibujar rectángulos
        for pt in zip(*ubicaciones[::-1]):  # Cambia las ubicaciones a (x, y)
            ubicaciones_coincidencias.append(pt)
            cv2.rectangle(imagen_principal, pt, (pt[0] + ancho_nuevo, pt[1] + alto_nuevo), (0, 255, 0), 2)

            # Mostrar la imagen con las coincidencias
        cv2.imshow('Coincidencias', imagen_principal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def temp3():
    import cv2

    # Cargar la imagen principal y la plantilla
    imagen_principal = cv2.imread(image_source)
    plantilla = cv2.imread(image_template)

    # Crear un objeto ORB
    orb = cv2.ORB_create()

    # Encontrar puntos clave y descriptores
    kp1, des1 = orb.detectAndCompute(plantilla, None)
    kp2, des2 = orb.detectAndCompute(imagen_principal, None)

    # Crear un objeto de coincidencia de Brute Force
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Coincidir los descriptores
    matches = bf.match(des1, des2)

    # Ordenar las coincidencias por distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Dibujar las mejores coincidencias
    imagen_matches = cv2.drawMatches(plantilla, kp1, imagen_principal, kp2, matches[:20], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Mostrar la imagen de coincidencias
    cv2.imshow('Coincidencias', imagen_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def temp4():
    import cv2
    import numpy as np

    # Cargar ambas imágenes
    imagen_principal = cv2.imread(image_source)
    imagen_template = cv2.imread(image_template)

    # Convertir ambas imágenes a escala de grises
    gris_principal = cv2.cvtColor(imagen_principal, cv2.COLOR_BGR2GRAY)
    gris_template = cv2.cvtColor(imagen_template, cv2.COLOR_BGR2GRAY)

    # Detectar bordes en la imagen de la plantilla
    #bordes_template = cv2.Canny(gris_template, 110, 255)

    # Encontrar contornos en la plantilla
    #contornos_template, _ = cv2.findContours(bordes_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contornos_template)

    bordes_template = findEdgesFromImage(image_template, False)
    contornos_template = findContourFromImage(bordes_template)
    drawContours(image_template, contornos_template, 3)

    # Solo tomar el primer contorno encontrado
    contorno_template = contornos_template[7]

    print(len(contornos_template))

    drawContours(image_template, contorno_template,3)

    # Para cada contorno encontrado, se ajusta un rectángulo y se busca en la imagen principal
    x, y, w, h = cv2.boundingRect(contorno_template)
    region_template = bordes_template[y:y + h, x:x + w]

    # Buscar el contorno similar en la imagen principal
    resultados = cv2.matchTemplate(gris_principal, region_template, cv2.TM_CCOEFF_NORMED)

    print('resultados' + str(resultados))

    umbral = 0.7
    ubicaciones = np.where(resultados >= umbral)

    # Dibujar rectángulos alrededor de las coincidencias en la imagen principal
    for pt in zip(*ubicaciones[::-1]):  # Cambiar ubicaciones a (x, y)
        cv2.rectangle(imagen_principal, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

        # Mostrar las imágenes
    cv2.imshow('Imagen Principal con Contornos Encontrados', imagen_principal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def temp5():

    # Cargar la imagen
    imagen = cv2.imread(image_source)

    # Obtener las dimensiones de la imagen
    alto, ancho = imagen.shape[:2]

    # Definir la región central (puedes ajustar el tamaño de la región)
    margen = 470  # Margen desde los bordes
    region_central = imagen[margen:alto - margen - 60, margen:ancho - margen + 10]

    # Convertir la región central a escala de grises
    gris_region = cv2.cvtColor(region_central, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque (opcional, pero recomendado para reducir ruido)
    gris_region = cv2.GaussianBlur(gris_region, (5, 5), 0)

    # Detectar bordes usando Canny
    bordes = cv2.Canny(gris_region, 0, 255)

    # Encontrar los contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original (opcional)
    imagen_con_contornos = imagen.copy()
    cv2.drawContours(imagen_con_contornos, contornos, -1, (0, 255, 0), 2)

    # Mostrar la región central y la imagen con contornos dibujados
    cv2.imshow('Región Central', region_central)
    cv2.imshow('Contornos en Imagen Original', imagen_con_contornos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""





"""
image = extraerCentro(image)

edge = findEdgesFromImage(image,True)
contours = findContourFromImage(edge,True)

print(cv2.contourArea(contours))
showCentroyGrados(image,contours)

#drawContours(image,contours,3)
image = cv2.imread('src\\assets\\images\\tear.jpg',0)
image2 = cv2.imread('src\\assets\\images\\image1.jpg',0)

encontrarSimilitud(image,image2)
"""


# print(len(contours))
# l = [contours[27],contours[38]]


# print(11, contours[11], cv2.contourArea(contours[11]))
# drawContours(image, contours[11],3)


# ----------------------------------------------------------------------------------------
# Deteccion de contornos con Eliminacion de Ruido y Suavizado de Contornos



#---------------------------------------------------------------------------------------------------------


    # Dibujar los contornos
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # Mostrar la imagen con los contornos
    # cv2.imshow('Contornos', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




