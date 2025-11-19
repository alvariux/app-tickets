import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

def mejorar_imagen_ocr(ruta_entrada, ruta_salida):
    """
    Mejora una imagen para optimizar el reconocimiento de texto OCR
    """
    # Cargar imagen
    img = cv2.imread(ruta_entrada)
    
    # Paso 1: Convertir a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Paso 2: Aplicar desenfoque gaussiano suave
    desenfoque = cv2.GaussianBlur(gris, (3, 3), 0)
    
    # Paso 3: Umbralización adaptativa
    umbral = cv2.adaptiveThreshold(
        desenfoque, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Paso 4: Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contraste = clahe.apply(umbral)
    
    # Paso 5: Reducción de ruido
    sin_ruido = cv2.medianBlur(contraste, 3)
    
    # Paso 6: Operaciones morfológicas para mejorar texto
    kernel = np.ones((1, 1), np.uint8)
    procesada = cv2.morphologyEx(sin_ruido, cv2.MORPH_CLOSE, kernel)
    procesada = cv2.morphologyEx(procesada, cv2.MORPH_OPEN, kernel)
    
    # Guardar imagen procesada
    cv2.imwrite(ruta_salida, procesada)
    
    return procesada

def mejorar_imagen_pil(ruta_entrada, ruta_salida):
    """
    Alternativa usando PIL para mejorar la imagen
    """
    # Abrir imagen con PIL
    img = Image.open(ruta_entrada)
    
    # Convertir a escala de grises
    gris = img.convert('L')
    
    # Mejorar contraste
    enhancer = ImageEnhance.Contrast(gris)
    contraste = enhancer.enhance(2.0)  # Aumentar contraste
    
    # Mejorar nitidez
    enhancer = ImageEnhance.Sharpness(contraste)
    nitida = enhancer.enhance(2.0)
    
    # Aplicar filtro para reducir ruido
    filtrada = nitida.filter(ImageFilter.MedianFilter(size=3))
    
    # Guardar imagen mejorada
    filtrada.save(ruta_salida, 'PNG', dpi=(300, 300))
    
    return filtrada

def preprocesamiento_completo(ruta_entrada, ruta_salida):
    """
    Procesamiento completo combinando OpenCV y PIL
    """
    # Usar OpenCV para procesamiento inicial
    img_cv = mejorar_imagen_ocr(ruta_entrada, "temp_processed.jpg")
    
    # Usar PIL para ajustes finales
    img_pil = mejorar_imagen_pil("temp_processed.jpg", ruta_salida)
    
    return img_pil

def mostrar_comparacion(ruta_original, ruta_procesada):
    """
    Mostrar comparación antes/después
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Imagen original
    img_orig = cv2.imread(ruta_original)
    img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_orig_rgb)
    ax1.set_title('Imagen Original')
    ax1.axis('off')
    
    # Imagen procesada
    img_proc = cv2.imread(ruta_procesada, cv2.IMREAD_GRAYSCALE)
    ax2.imshow(img_proc, cmap='gray')
    ax2.set_title('Imagen Mejorada para OCR')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def extraer_texto_ocr(ruta_imagen):
    """
    Extraer texto usando pytesseract (requiere instalación)
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Cargar imagen procesada
        img = Image.open(ruta_imagen)
        
        # Configurar parámetros para OCR
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúÑñ.,$- /kg'
        
        # Extraer texto
        texto = pytesseract.image_to_string(img, config=config, lang='spa')
        
        return texto
    
    except ImportError:
        return "pytesseract no está instalado. Instala con: pip install pytesseract"

# USO DEL CÓDIGO
if __name__ == "__main__":
    # Configurar rutas
    ruta_imagen_original = "img/IMG_0758_c.jpg"  # Cambia por tu ruta
    ruta_imagen_mejorada = "imagen_mejorada_ocr.png"
    
    # Procesar imagen
    print("Procesando imagen...")
    imagen_procesada = preprocesamiento_completo(ruta_imagen_original, ruta_imagen_mejorada)
    
    # Mostrar comparación
    mostrar_comparacion(ruta_imagen_original, ruta_imagen_mejorada)
    
    print(f"Imagen mejorada guardada como: {ruta_imagen_mejorada}")
    
    # Intentar extraer texto (si pytesseract está instalado)
    texto_extraido = extraer_texto_ocr(ruta_imagen_mejorada)
    print("\nTexto extraído:")
    print(texto_extraido)

# Código adicional para procesamiento específico de tickets
def procesar_ticket_mercado(ruta_imagen):
    """
    Procesamiento especializado para tickets de supermercado
    """
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    
    # Redimensionar si es muy grande
    altura, ancho = img.shape[:2]
    if altura > 2000:
        escala = 2000 / altura
        nuevo_ancho = int(ancho * escala)
        img = cv2.resize(img, (nuevo_ancho, 2000))
    
    # Convertir a grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enfocar la imagen
    kernel_enfoque = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enfocada = cv2.filter2D(gris, -1, kernel_enfoque)
    
    # Umbralización adaptativa para texto
    umbral = cv2.adaptiveThreshold(
        enfocada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 10
    )
    
    # Guardar resultado
    cv2.imwrite("ticket_procesado.png", umbral)
    
    return umbral

# Ejemplo de uso específico para tu ticket
def procesar_tu_ticket():
    """
    Procesa específicamente el ticket que compartiste
    """
    # Usar el procesamiento especializado
    imagen_procesada = procesar_ticket_mercado("IMG_0758_c.jpg")
    
    # Guardar resultado
    cv2.imwrite("ticket_smart_mejorado.png", imagen_procesada)
    print("Ticket procesado y guardado como 'ticket_smart_mejorado.png'")

# Para instalar las dependencias necesarias:
def instrucciones_instalacion():
    """
    Muestra las instrucciones para instalar las dependencias
    """
    print("""
    DEPENDENCIAS NECESARIAS:
    
    pip install opencv-python
    pip install pillow
    pip install matplotlib
    pip install numpy
    pip install pytesseract
    
    ADEMÁS:
    - Instalar Tesseract OCR desde: https://github.com/UB-Mannheim/tesseract/wiki
    - Asegurarse de que esté en el PATH del sistema
    """)

# Ejecutar esta función para ver las instrucciones de instalación
# instrucciones_instalacion()