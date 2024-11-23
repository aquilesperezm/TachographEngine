import numpy as np
import cv2
from math import pi
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os


class AplicacionTacografo:
    def __init__(self, root):
        self.root = root
        self.root.title("Transformador de Tacógrafos")
        self.root.geometry("800x600")

        self.ruta_imagen = None
        self.imagen_original = None
        self.imagen_transformada = None

        self.crear_interfaz()

    def crear_interfaz(self):
        self.frame_principal = ttk.Frame(self.root, padding="10")
        self.frame_principal.grid(row=0, column=0, sticky="nsew")

        frame_botones = ttk.Frame(self.frame_principal)
        frame_botones.grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Button(
            frame_botones,
            text="Seleccionar Imagen",
            command=self.seleccionar_imagen,
        ).grid(row=0, column=0, padx=5)
        ttk.Button(
            frame_botones,
            text="Transformar",
            command=self.procesar_imagen,
        ).grid(row=0, column=1, padx=5)
        ttk.Button(
            frame_botones,
            text="Guardar Resultado",
            command=self.guardar_resultado,
        ).grid(row=0, column=2, padx=5)

        ttk.Label(self.frame_principal, text="Imagen Original").grid(row=1, column=0)
        ttk.Label(self.frame_principal, text="Imagen Transformada").grid(
            row=1, column=1
        )

        self.canvas_original = tk.Canvas(self.frame_principal, width=350, height=350)
        self.canvas_original.grid(row=2, column=0, padx=5)

        self.canvas_transformada = tk.Canvas(self.frame_principal, width=350, height=350)
        self.canvas_transformada.grid(row=2, column=1, padx=5)

        self.estado = ttk.Label(
            self.frame_principal, text="Esperando imagen..."
        )
        self.estado.grid(row=3, column=0, columnspan=2, pady=5)

    def seleccionar_imagen(self):
        self.ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar imagen de tacógrafo",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")],
        )
        if self.ruta_imagen:
            self.imagen_original = cv2.imread(self.ruta_imagen)
            if self.imagen_original is not None:
                self.mostrar_imagen(self.canvas_original, self.imagen_original)
                self.estado.config(
                    text="Imagen cargada: " + os.path.basename(self.ruta_imagen)
                )

    def mostrar_imagen(self, canvas, imagen):
        if imagen is None:
            return

        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        altura, ancho = imagen.shape[:2]
        max_dim = 350

        if altura > ancho:
            nuevo_alto = min(altura, max_dim)
            factor = nuevo_alto / altura
            nuevo_ancho = int(ancho * factor)
        else:
            nuevo_ancho = min(ancho, max_dim)
            factor = nuevo_ancho / ancho
            nuevo_alto = int(altura * factor)

        imagen_redim = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
        imagen_tk = ImageTk.PhotoImage(Image.fromarray(imagen_redim))
        canvas.imagen_tk = imagen_tk
        canvas.create_image(max_dim // 2, max_dim // 2, image=imagen_tk)

    def transformar_a_rectangular(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (9, 9), 2)

        # Detectar el círculo más grande
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=int(min(imagen.shape[:2]) * 0.3),
            maxRadius=int(min(imagen.shape[:2]) * 0.5),
        )

        if circulos is None:
            raise Exception("No se detectó ningún círculo en la imagen")

        centro_x, centro_y, radio = np.round(circulos[0][0]).astype(int)

        # Ampliar el radio para incluir áreas cercanas al borde
        radio += 10  # Incrementar el radio según sea necesario

        alto = int(radio)
        ancho = 1440  # 4 píxeles por grado
        rectangular = np.zeros((alto, ancho, 3), dtype=np.uint8)

        for y in range(alto):
            for x in range(ancho):
                theta = (2 * pi * x) / ancho
                r = radio * (1 - y / alto)

                pos_x = int(centro_x + r * np.cos(theta))
                pos_y = int(centro_y + r * np.sin(theta))

                if (0 <= pos_x < imagen.shape[1] and 0 <= pos_y < imagen.shape[0]):
                    rectangular[y, x] = imagen[pos_y, pos_x]

        return rectangular

    def encontrar_posicion_por_patron(self, imagen_rect):
        """
        Detecta la posición inicial de las 12 horas basándose en la línea negra y el patrón numérico (12, 13, 14).
        """
        # Paso 1: Detectar la línea negra en la parte superior
        franja_superior = imagen_rect[5:80, :]  # Ajustar según sea necesario
        gris = cv2.cvtColor(franja_superior, cv2.COLOR_BGR2GRAY)
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Detectar contornos para encontrar la línea negra más prominente
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_ancho = 0
        pos_x_linea = None

        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            if w > max_ancho and h > 10:  # Validar altura mínima para evitar ruido
                max_ancho = w
                pos_x_linea = x

        if pos_x_linea is None:
            raise Exception("No se pudo detectar la línea negra superior")

        # Paso 2: Buscar el patrón "12, 13, 14" justo debajo de la línea negra
        franja_inferior = imagen_rect[80:160, :]  # Ampliar el rango para incluir los números
        gris_inferior = cv2.cvtColor(franja_inferior, cv2.COLOR_BGR2GRAY)

        # Aplicar binarización para resaltar los números
        _, binaria_inferior = cv2.threshold(gris_inferior, 128, 255, cv2.THRESH_BINARY_INV)

        # Buscar contornos en la franja inferior
        contornos_inferior, _ = cv2.findContours(binaria_inferior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        numeros_posiciones = []

        for contorno in contornos_inferior:
            x, y, w, h = cv2.boundingRect(contorno)
            if h > 15 and w > 10:  # Validar tamaño mínimo de los números
                numeros_posiciones.append((x, w))

        # Ordenar posiciones de izquierda a derecha
        numeros_posiciones.sort(key=lambda pos: pos[0])

        # Buscar el patrón "12, 13, 14"
        for i in range(len(numeros_posiciones) - 2):
            x1, w1 = numeros_posiciones[i]
            x2, w2 = numeros_posiciones[i + 1]
            x3, w3 = numeros_posiciones[i + 2]

            if (x2 - x1 > 30 and x3 - x2 > 30):  # Validar separación típica entre números
                print(f"Patrón '12, 13, 14' encontrado en posición {x1}")
                return x1

        raise Exception("No se pudo detectar el patrón '12, 13, 14'")

    def procesar_imagen(self):
        """
        Procesa la imagen seleccionada, transformándola a rectangular y alineándola.
        """
        try:
            if self.imagen_original is None:
                raise Exception("Por favor, seleccione una imagen primero")

            # Transformar la imagen circular a rectangular
            rectangular = self.transformar_a_rectangular(self.imagen_original)

            # Detectar la posición de las 12 horas
            pos_12 = self.encontrar_posicion_por_patron(rectangular)

            # Ajustar la imagen para alinear las 12 horas al inicio del eje X
            self.imagen_transformada = np.roll(rectangular, -pos_12, axis=1)

            # Dibujar líneas de guía para facilitar la visualización
            for x in range(0, self.imagen_transformada.shape[1], 60):
                cv2.line(self.imagen_transformada, (x, 0),
                         (x, self.imagen_transformada.shape[0]), (255, 0, 0), 1)
                cv2.putText(self.imagen_transformada, str(x), (x, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Mostrar la imagen transformada en la interfaz
            self.mostrar_imagen(self.canvas_transformada, self.imagen_transformada)
            self.estado.config(text="Transformación completada")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.estado.config(text="Error en la transformación")

    def guardar_resultado(self):
        if self.imagen_transformada is None:
            messagebox.showerror("Error", "No hay imagen transformada para guardar")
            return

        ruta_guardar = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos los archivos", "*.*")],
        )

        if ruta_guardar:
            cv2.imwrite(ruta_guardar, self.imagen_transformada)
            self.estado.config(text=f"Imagen guardada en: {os.path.basename(ruta_guardar)}")


def main():
    root = tk.Tk()
    app = AplicacionTacografo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
