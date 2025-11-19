import cv2
import numpy as np
import imutils
import argparse

def procesar_imagen_real(ruta_imagen):
    """
    Procesamiento REAL para tu imagen de ticket de gasolina
    """    
    img = cv2.imread(ruta_imagen)
    
    # Paso 1: Redimensionar para mejor calidad OCR
    altura, ancho = img.shape[:2]    
    
    # Reducir si es muy grande, aumentar si es muy pequeña
    if altura > 1500:
        factor = 1500 / altura
        nuevo_ancho = int(ancho * factor)
        img = cv2.resize(img, (nuevo_ancho, 1500), interpolation=cv2.INTER_CUBIC)
    
    # Paso 2: Convertir a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Paso 3: Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contraste = clahe.apply(gris)

    #return img, contraste
    
    # Paso 4: Aplicar filtro bilateral para reducir ruido manteniendo bordes
    filtrado = cv2.bilateralFilter(contraste, 9, 75, 75)

    #return img, filtrado
    
    # Paso 5: Umbralización adaptativa para texto
    #umbral = cv2.adaptiveThreshold(
    #    filtrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #    cv2.THRESH_BINARY, 15, 12
    #)
    
    # Usar filtro bilateral para preservar bordes
    suavizado = cv2.bilateralFilter(gris, 15, 80, 80)    

    # 4. Umbralización adaptativa MÁS CONSERVADORA
    umbral = cv2.adaptiveThreshold(
        suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 8  # Parámetros más conservadores
    )

    
    # Paso 6: Operaciones morfológicas para limpiar
    #kernel = np.ones((2, 2), np.uint8)
    #procesada = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
    #procesada = cv2.morphologyEx(procesada, cv2.MORPH_OPEN, kernel)

    # Solo limpieza mínima
    kernel = np.ones((1, 1), np.uint8)
    procesada = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)
    
    return img, procesada


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be processed")

args = vars(ap.parse_args())

img, img_procesada = procesar_imagen_real(args["image"])
cv2.imwrite("imagen_original_real.png", img)
cv2.imwrite("imagen_procesada_real.png", img_procesada)