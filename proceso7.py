import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

def procesar_imagen_real(ruta_imagen):
    """
    Procesamiento REAL para tu imagen de ticket de gasolina
    """
    print("🔍 Cargando imagen...")
    img = cv2.imread(ruta_imagen)
    
    # Paso 1: Redimensionar para mejor calidad OCR
    altura, ancho = img.shape[:2]
    print(f"📐 Tamaño original: {ancho}x{altura}")
    
    # Reducir si es muy grande, aumentar si es muy pequeña
    if altura > 1500:
        factor = 1500 / altura
        nuevo_ancho = int(ancho * factor)
        img = cv2.resize(img, (nuevo_ancho, 1500), interpolation=cv2.INTER_CUBIC)
        print("📏 Imagen redimensionada a 1500px de altura")
    
    # Paso 2: Convertir a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Paso 3: Mejorar contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contraste = clahe.apply(gris)
    
    # Paso 4: Aplicar filtro bilateral para reducir ruido manteniendo bordes
    filtrado = cv2.bilateralFilter(contraste, 9, 75, 75)
    
    # Paso 5: Umbralización adaptativa para texto
    umbral = cv2.adaptiveThreshold(
        filtrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 12
    )
    
    # Paso 6: Operaciones morfológicas para limpiar
    kernel = np.ones((2, 2), np.uint8)
    procesada = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
    procesada = cv2.morphologyEx(procesada, cv2.MORPH_OPEN, kernel)
    
    return img, procesada

def extraer_texto_con_configuraciones(imagen_procesada):
    """
    Probar múltiples configuraciones de OCR
    """
    configuraciones = [
        {
            "nombre": "Configuración Estándar",
            "config": "--oem 3 --psm 6 -l spa"
        },
        {
            "nombre": "Configuración Ticket", 
            "config": "--oem 3 --psm 6 -l spa -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚáéíóúÑñ.,$/- :()"
        },
        {
            "nombre": "Configuración Una Línea",
            "config": "--oem 3 --psm 7 -l spa"
        },
        {
            "nombre": "Configuración Bloque Único",
            "config": "--oem 3 --psm 8 -l spa"
        }
    ]
    
    resultados = []
    
    for config in configuraciones:
        print(f"🔄 Probando: {config['nombre']}")
        try:
            texto = pytesseract.image_to_string(imagen_procesada, config=config['config'])
            resultados.append((config['nombre'], texto))
        except Exception as e:
            print(f"❌ Error con {config['nombre']}: {e}")
    
    return resultados

# PROCESAMIENTO DE TU IMAGEN
if __name__ == "__main__":
    # Procesar la imagen
    img_original, img_procesada = procesar_imagen_real("img/IMG_0759.jpg")
    
    # Guardar imagen procesada
    cv2.imwrite("imagen_procesada_real.png", img_procesada)
    print("💾 Imagen procesada guardada como: imagen_procesada_real.png")
    
    # Extraer texto con diferentes configuraciones
    print("\n🧪 Extrayendo texto con diferentes configuraciones...")
    resultados = extraer_texto_con_configuraciones(img_procesada)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS REALES DEL OCR")
    print("="*60)
    
    for i, (nombre, texto) in enumerate(resultados, 1):
        print(f"\n📄 {i}. {nombre}:")
        print("-" * 40)
        print(texto)
        print("-" * 40)
    
    # Guardar el mejor resultado (usualmente el primero)
    if resultados:
        mejor_resultado = resultados[0][1]  # Tomar el primero
        with open("texto_extraido_real.txt", "w", encoding="utf-8") as f:
            f.write(mejor_resultado)
        print(f"\n💾 Texto guardado en: texto_extraido_real.txt")

# ANÁLISIS ADICIONAL PARA MEJORAR PRECISIÓN
def analizar_imagen_para_mejoras(ruta_imagen):
    """
    Analiza la imagen para sugerir mejoras específicas
    """
    img = cv2.imread(ruta_imagen)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcular métricas de calidad
    brillo_medio = np.mean(gris)
    contraste = np.std(gris)
    
    print(f"\n📊 ANÁLISIS DE LA IMAGEN:")
    print(f"   Brillo medio: {brillo_medio:.2f}")
    print(f"   Contraste: {contraste:.2f}")
    
    if brillo_medio < 100:
        print("   💡 Recomendación: Aumentar brillo")
    if contraste < 50:
        print("   🎨 Recomendación: Mejorar contraste")
    
    # Detectar orientación
    try:
        osd = pytesseract.image_to_osd(img)
        print(f"   🧭 Orientación detectada: {osd}")
    except:
        print("   🧭 No se pudo detectar orientación automáticamente")

# Ejecutar análisis
analizar_imagen_para_mejoras("img/IMG_0759.jpg")