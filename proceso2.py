import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import sys
import os
import imutils

def eliminar_lineas_imagen(imagen):    
    """
    Método morfológico: Detecta y elimina líneas horizontales largas.
    MEJOR PARA: Subrayados continuos y gruesos.
    """
    # Invertir imagen para operaciones morfológicas
    invertida = cv2.bitwise_not(imagen)
    cv2.imwrite("invertida.png", invertida)
    
    # Crear kernel horizontal largo para detectar líneas
    ancho_img = imagen.shape[1]
    longitud_linea = ancho_img // 30  # Líneas de al menos 1/30 del ancho
    
    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT, 
        (longitud_linea, 1)
    )
    
    # Detectar líneas horizontales
    lineas_detectadas = cv2.morphologyEx(
        invertida, 
        cv2.MORPH_OPEN, 
        kernel_horizontal, 
        iterations=2
    )
    
    # Dilatar las líneas detectadas para cubrir completamente
    kernel_dilatar = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lineas_dilatadas = cv2.dilate(lineas_detectadas, kernel_dilatar, iterations=1)
    
    # Crear máscara de las líneas
    _, mascara = cv2.threshold(lineas_dilatadas, 0, 255, cv2.THRESH_BINARY)
    
    # Eliminar las líneas usando inpainting
    resultado = cv2.inpaint(imagen, mascara, 3, cv2.INPAINT_TELEA)
    
    return resultado

def eliminar_subrayados_hough(imagen):
    """
    Método de Transformada de Hough: Detecta líneas con precisión matemática.
    MEJOR PARA: Subrayados rectos y bien definidos.
    """
    # Detectar bordes
    bordes = cv2.Canny(imagen, 50, 150, apertureSize=3)
    
    # Detectar líneas con Transformada de Hough
    lineas = cv2.HoughLinesP(
        bordes,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=imagen.shape[1]//4,  # Al menos 1/4 del ancho
        maxLineGap=10
    )
    
    # Crear máscara para las líneas
    mascara = np.zeros(imagen.shape, dtype=np.uint8)
    
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            
            # Filtrar solo líneas horizontales (ángulo pequeño)
            angulo = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angulo < 10 or angulo > 170:  # Líneas casi horizontales
                # Dibujar línea gruesa en la máscara
                cv2.line(mascara, (x1, y1), (x2, y2), 255, 5)
    
    # Aplicar inpainting si se detectaron líneas
    if np.any(mascara):
        resultado = cv2.inpaint(imagen, mascara, 3, cv2.INPAINT_TELEA)
    else:
        resultado = imagen
    
    return resultado


def eliminar_subrayados_proyeccion(imagen):
    """
    Método de proyección horizontal: Detecta líneas por densidad de píxeles.
    MEJOR PARA: Subrayados finos o discontinuos.
    """
    # Binarizar si no está binarizada
    if len(np.unique(imagen)) > 2:
        _, binaria = cv2.threshold(imagen, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binaria = imagen.copy()
    
    # Invertir para que el texto sea blanco
    invertida = cv2.bitwise_not(binaria)
    
    # Calcular proyección horizontal (suma de píxeles por fila)
    proyeccion = np.sum(invertida, axis=1)
    
    # Normalizar
    proyeccion = proyeccion / imagen.shape[1]
    
    # Detectar filas con alta densidad (posibles líneas)
    umbral = np.mean(proyeccion) + 1.5 * np.std(proyeccion)
    filas_lineas = np.where(proyeccion > umbral)[0]
    
    # Crear máscara para líneas detectadas
    mascara = np.zeros(imagen.shape, dtype=np.uint8)
    
    for fila in filas_lineas:
        # Verificar que sea una línea y no texto
        fila_pixels = invertida[fila, :]
        
        # Si más del 60% de la fila tiene píxeles, es probable que sea una línea
        if np.sum(fila_pixels > 0) > imagen.shape[1] * 0.6:
            mascara[max(0, fila-2):min(imagen.shape[0], fila+3), :] = 255
    
    # Aplicar inpainting
    if np.any(mascara):
        resultado = cv2.inpaint(imagen, mascara, 3, cv2.INPAINT_TELEA)
    else:
        resultado = imagen
    
    return resultado

def desenfoque_gaussiano(imagen, ksize=(5, 5), sigma=0):
    """Aplica un desenfoque gaussiano a la imagen."""
    return cv2.GaussianBlur(imagen, ksize, sigma)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagen", required=True,
        help="Ruta a la imagen de entrada")
    args = vars(ap.parse_args())

    # cargar la imagen y convertirla a escala de grises
    imagen = cv2.imread(args["imagen"])
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray_original.png", gray)
    
    # eliminar líneas de la imagen
    gray = eliminar_subrayados_proyeccion(gray)
    cv2.imwrite("sin_lineas.png", gray)
    
    # aplicar desenfoque gaussiano
    gray = desenfoque_gaussiano(gray)
    cv2.imwrite("desenfocado.png", gray)


    