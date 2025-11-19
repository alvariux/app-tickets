import cv2
import imutils
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be processed")

def orient_vertical(img):
    width = img.shape[1]
    height = img.shape[0]
  
    if width > height:
        rotated = imutils.rotate(img, angle=270)
    else:
        rotated = img
    return rotated

def sharpen_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dilated = cv2.dilate(blurred, rectKernel, iterations=2)
    edged = cv2.Canny(dilated, 75, 200, apertureSize=3)
    return edged

def binarize(img, threshold):
    threshold = np.mean(img)
    thresh, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(binary, rectKernel, iterations=2)
    return dilated

def find_receipt_bounding_box(binary, img):
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_cnt)
    box = np.intp(cv2.boxPoints(rect))
    boxed = cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 20)
    return boxed, largest_cnt, rect

def find_tilt_angle(rect):
    angle = rect[2]  # Find the angle of vertical line
    print("Angle_0 = ", round(angle, 1))
    
    uniform_angle = angle
    if angle < -45:
        angle += 90
        print("Angle_1:", round(angle, 1))
        uniform_angle = abs(angle)
    else:
        uniform_angle = abs(angle)
    
    print("Uniform angle = ", round(uniform_angle, 1))
    return uniform_angle

def adjust_tilt(img, angle):
    if angle >= 5 and angle < 80:
        rotated_angle = 0
    elif angle < 5:
        rotated_angle = angle
    else:
        rotated_angle = 270 + angle
    
    tilt_adjusted = imutils.rotate(img, rotated_angle)
    delta = 360 - rotated_angle
    return tilt_adjusted, delta

def crop(img, largest_contour):
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y+h, x:x+w]
    return cropped

def enhance_txt(img):
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w * 0.05)
    w2 = int(w * 0.95)
    h1 = int(h * 0.05)
    h2 = int(h * 0.95)
    ROI = img[h1:h2, w1:w2]  # 95% of center of the image
    threshold = np.mean(ROI) * 0.98  # % of average brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)
    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return binary

# Programa principal
args = vars(ap.parse_args())
raw_img = cv2.imread(args["image"])

# Orientar verticalmente
rotated = orient_vertical(raw_img)
cv2.imwrite("rotated.png", rotated)

# Detectar bordes
edged = sharpen_edge(rotated)
cv2.imwrite("edged.png", edged)

# Binarizar
threshold = 100
binary = binarize(edged, threshold)
cv2.imwrite("binarized.png", binary)

# Encontrar el contorno del recibo
boxed, largest_cnt, rect = find_receipt_bounding_box(binary, rotated)
boxed_rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)

# Encontrar ángulo de inclinación
angle = find_tilt_angle(rect)

# Ajustar inclinación
tilted, delta = adjust_tilt(boxed, angle)
tilted_rgb = cv2.cvtColor(tilted, cv2.COLOR_BGR2RGB)

# Recortar
cropped = crop(tilted, largest_cnt)
cv2.imwrite("cropped.png", cropped)

# Mejorar texto
enhanced = enhance_txt(cropped)
enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
cv2.imwrite("enhanced.png", enhanced)