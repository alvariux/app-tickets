import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import sys
import os
import imutils




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagen", required=True,
        help="Ruta a la imagen de entrada")
    args = vars(ap.parse_args())

    # cargar la imagen y convertirla a escala de grises
    imagen = cv2.imread(args["imagen"])
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray_original.png", gray)

    # aplicar un umbral binario inverso + Otsu para obtener una imagen binarizada
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite("thresh_original.png", thresh)

    # morphological transformations para eliminar ruido y separar caracteres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite("opening_original.png", opening)
    
    # encontrar contornos de los caracteres
    cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    chars = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 15 and h >= 30:
            chars.append((x, y, w, h))
    chars = sorted(chars, key=lambda b: b[0])
    mask = np.zeros(imagen.shape[:2], dtype="uint8")
    for (x, y, w, h) in chars:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    cv2.imwrite("mask_original.png", mask)
    final = cv2.bitwise_and(opening, opening, mask=mask)
    cv2.imwrite("final_original.png", final)

    
    


    