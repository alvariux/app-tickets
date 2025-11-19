import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import sys
import os

def mejorar_imagen_recibo(imagen_path, guardar_pasos=False, recortar=True,
                          resolucion_min=1000, nivel_ruido=10, 
                          factor_contraste=2.0, kernel_binarizacion=11):
    """
    Mejora una imagen de recibo para OCR óptimo.
    
    Parámetros:
    -----------
    imagen_path : str
        Ruta de la imagen del recibo
    guardar_pasos : bool
        Si es True, guarda imágenes intermedias del proceso
    recortar : bool
        Si es True, detecta y recorta automáticamente el recibo del fondo
    resolucion_min : int
        Altura mínima en píxeles para redimensionar (default: 1000)
    nivel_ruido : int
        Parámetro h para eliminación de ruido (1-20, default: 10)
    factor_contraste : float
        Factor para CLAHE (default: 2.0)
    kernel_binarizacion : int
        Tamaño del kernel para binarización adaptativa (default: 11)
    
    Retorna:
    --------
    imagen_mejorada : numpy.ndarray
        Imagen procesada lista para OCR
    """
    
    # 1. CARGAR IMAGEN
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    original = img.copy()
    
    # 1.5. RECORTAR RECIBO DEL FONDO
    if recortar:
        img = detectar_y_recortar_recibo(img)
        if guardar_pasos:
            cv2.imwrite('paso0_recortado.jpg', img)
    
    # 2. CORREGIR ORIENTACIÓN
    img_rotada = corregir_orientacion(img)
    if guardar_pasos:
        cv2.imwrite('paso1_orientacion.jpg', img_rotada)
    
    # 3. CONVERTIR A ESCALA DE GRISES
    gris = cv2.cvtColor(img_rotada, cv2.COLOR_BGR2GRAY)
    if guardar_pasos:
        cv2.imwrite('paso2_grises.jpg', gris)
    
    # 4. ELIMINAR RUIDO (Filtro bilateral preserva bordes)
    sin_ruido = cv2.fastNlMeansDenoising(gris, None, h=nivel_ruido, 
                                          templateWindowSize=7, 
                                          searchWindowSize=21)
    if guardar_pasos:
        cv2.imwrite('paso3_sin_ruido.jpg', sin_ruido)
    
    # 5. AUMENTAR CONTRASTE (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=factor_contraste, tileGridSize=(8,8))
    contraste = clahe.apply(sin_ruido)
    if guardar_pasos:
        cv2.imwrite('paso4_contraste.jpg', contraste)
    
    # 6. BINARIZACIÓN ADAPTATIVA (mejor para iluminación irregular)
    binaria = cv2.adaptiveThreshold(
        contraste, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        kernel_binarizacion, 
        2
    )
    if guardar_pasos:
        cv2.imwrite('paso5_binarizacion.jpg', binaria)
    
    # 7. OPERACIONES MORFOLÓGICAS (reconstruir texto)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    
    # Eliminar pequeños puntos de ruido
    apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Cerrar pequeños huecos en las letras
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel_cierre, iterations=1)
    
    if guardar_pasos:
        cv2.imwrite('paso6_morfologia.jpg', cierre)
    
    # 8. REDIMENSIONAR SI ES NECESARIO (OCR funciona mejor con resolución óptima)
    altura, ancho = cierre.shape
    if altura < resolucion_min:
        factor = resolucion_min / altura
        nueva_ancho = int(ancho * factor)
        cierre = cv2.resize(cierre, (nueva_ancho, resolucion_min), 
                           interpolation=cv2.INTER_CUBIC)
    
    if guardar_pasos:
        cv2.imwrite('paso7_final.jpg', cierre)
    
    return cierre


def corregir_orientacion(imagen):
    """
    Detecta y corrige la orientación del texto en la imagen.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Probar con diferentes ángulos de rotación
    mejor_confianza = 0
    mejor_angulo = 0
    mejor_imagen = imagen.copy()
    
    # Tesseract puede detectar orientación
    try:
        # Detectar orientación con Tesseract
        osd = pytesseract.image_to_osd(gris)
        
        # Extraer ángulo de rotación
        angulo = int(osd.split('Rotate: ')[1].split('\n')[0])
        
        if angulo != 0:
            # Rotar imagen
            (h, w) = imagen.shape[:2]
            centro = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
            mejor_imagen = cv2.warpAffine(imagen, M, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
    except:
        # Si falla la detección automática, probar rotaciones comunes
        for angulo in [0, 90, 180, 270]:
            if angulo == 0:
                img_rotada = imagen.copy()
            else:
                img_rotada = rotar_imagen(imagen, angulo)
            
            # Evaluar calidad de texto con OCR
            gris_temp = cv2.cvtColor(img_rotada, cv2.COLOR_BGR2GRAY)
            try:
                datos = pytesseract.image_to_data(gris_temp, output_type=pytesseract.Output.DICT)
                confianzas = [int(c) for c in datos['conf'] if c != '-1']
                if confianzas:
                    confianza_promedio = sum(confianzas) / len(confianzas)
                    if confianza_promedio > mejor_confianza:
                        mejor_confianza = confianza_promedio
                        mejor_angulo = angulo
                        mejor_imagen = img_rotada
            except:
                continue
    
    return mejor_imagen


def detectar_y_recortar_recibo(imagen):
    """
    Detecta automáticamente el recibo en la imagen y lo recorta,
    eliminando el fondo.
    
    Parámetros:
    -----------
    imagen : numpy.ndarray
        Imagen original con el recibo y fondo
    
    Retorna:
    --------
    recibo_recortado : numpy.ndarray
        Imagen con solo el recibo, sin fondo
    """
    original = imagen.copy()
    altura_orig, ancho_orig = imagen.shape[:2]
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reducir ruido
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Detección de bordes con Canny
    bordes = cv2.Canny(blur, 50, 150)
    
    # Dilatar para conectar bordes fragmentados
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilatado = cv2.dilate(bordes, kernel, iterations=2)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        print("⚠️  No se detectaron contornos. Retornando imagen original.")
        return original
    
    # Ordenar contornos por área (de mayor a menor)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    recibo_detectado = None
    
    # Buscar el contorno que probablemente sea el recibo
    for contorno in contornos[:5]:  # Revisar los 5 contornos más grandes
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        
        # Filtrar contornos muy pequeños o muy grandes
        area_minima = (ancho_orig * altura_orig) * 0.1  # Al menos 10% de la imagen
        area_maxima = (ancho_orig * altura_orig) * 0.95  # Máximo 95% de la imagen
        
        if area < area_minima or area > area_maxima:
            continue
        
        # Aproximar el contorno a un polígono
        epsilon = 0.02 * perimetro
        approx = cv2.approxPolyDP(contorno, epsilon, True)
        
        # Un recibo típicamente tiene 4 esquinas (rectángulo)
        if len(approx) >= 4:
            recibo_detectado = approx
            break
    
    # Si no se detectó un recibo con 4 esquinas, usar el contorno más grande
    if recibo_detectado is None:
        recibo_detectado = contornos[0]
        print("⚠️  No se detectó rectángulo. Usando contorno más grande.")
    
    # Obtener rectángulo delimitador
    x, y, w, h = cv2.boundingRect(recibo_detectado)
    
    # Aplicar márgenes pequeños para no cortar texto
    margen_x = int(w * 0.02)
    margen_y = int(h * 0.02)
    
    x = max(0, x - margen_x)
    y = max(0, y - margen_y)
    w = min(ancho_orig - x, w + 2 * margen_x)
    h = min(altura_orig - y, h + 2 * margen_y)
    
    # Verificar que el área recortada sea razonable
    area_recortada = w * h
    area_total = ancho_orig * altura_orig
    
    if area_recortada < area_total * 0.05:
        print("⚠️  Área detectada muy pequeña. Retornando imagen original.")
        return original
    
    # Recortar el recibo
    recibo_recortado = original[y:y+h, x:x+w]
    
    # Aplicar corrección de perspectiva si se detectaron 4 esquinas
    if len(recibo_detectado) == 4:
        recibo_recortado = corregir_perspectiva(original, recibo_detectado)
    
    print(f"✓ Recibo detectado y recortado: {w}x{h} píxeles")
    
    return recibo_recortado


def corregir_perspectiva(imagen, puntos):
    """
    Corrige la perspectiva del recibo si está inclinado o visto desde ángulo.
    
    Parámetros:
    -----------
    imagen : numpy.ndarray
        Imagen original
    puntos : numpy.ndarray
        Array de 4 puntos que definen las esquinas del recibo
    
    Retorna:
    --------
    imagen_corregida : numpy.ndarray
        Imagen con perspectiva corregida
    """
    # Ordenar puntos: arriba-izq, arriba-der, abajo-der, abajo-izq
    puntos = puntos.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # La suma de coordenadas identifica esquinas opuestas
    s = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(s)]  # Arriba-izquierda (suma más pequeña)
    rect[2] = puntos[np.argmax(s)]  # Abajo-derecha (suma más grande)
    
    # La diferencia identifica las otras esquinas
    diff = np.diff(puntos, axis=1)
    rect[1] = puntos[np.argmin(diff)]  # Arriba-derecha
    rect[3] = puntos[np.argmax(diff)]  # Abajo-izquierda
    
    # Calcular ancho y alto del recibo corregido
    (tl, tr, br, bl) = rect
    
    ancho_arriba = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    ancho_abajo = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    ancho_max = max(int(ancho_arriba), int(ancho_abajo))
    
    alto_izq = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    alto_der = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alto_max = max(int(alto_izq), int(alto_der))
    
    # Puntos de destino para la transformación
    dst = np.array([
        [0, 0],
        [ancho_max - 1, 0],
        [ancho_max - 1, alto_max - 1],
        [0, alto_max - 1]
    ], dtype="float32")
    
    # Calcular matriz de transformación de perspectiva
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Aplicar transformación
    warped = cv2.warpPerspective(imagen, M, (ancho_max, alto_max))
    
    return warped


def rotar_imagen(imagen, angulo):
    """
    Rota una imagen por un ángulo específico.
    """
    (h, w) = imagen.shape[:2]
    centro = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    
    # Calcular nuevas dimensiones
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nuevo_w = int((h * sin) + (w * cos))
    nuevo_h = int((h * cos) + (w * sin))
    
    # Ajustar matriz de rotación
    M[0, 2] += (nuevo_w / 2) - centro[0]
    M[1, 2] += (nuevo_h / 2) - centro[1]
    
    return cv2.warpAffine(imagen, M, (nuevo_w, nuevo_h), 
                          borderMode=cv2.BORDER_REPLICATE)


# EJEMPLO DE USO
if __name__ == "__main__":
    # Configurar argumentos de línea de comando
    parser = argparse.ArgumentParser(
        description='Mejora imágenes de recibos para OCR óptimo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s recibo.jpg
  %(prog)s recibo.jpg -o salida.jpg
  %(prog)s recibo.jpg --sin-recortar --guardar-pasos
  %(prog)s recibo.jpg --idioma eng --salida-texto resultado.txt
  %(prog)s recibo.jpg -r 300 -c 3.0 --kernel 15
        """
    )
    
    # Argumentos posicionales
    parser.add_argument('imagen', 
                        help='Ruta de la imagen del recibo a procesar')
    
    # Argumentos opcionales - Salida
    parser.add_argument('-o', '--salida-imagen', 
                        default='recibo_mejorado.jpg',
                        help='Ruta donde guardar la imagen mejorada (default: recibo_mejorado.jpg)')
    
    parser.add_argument('-t', '--salida-texto',
                        default='texto_extraido.txt',
                        help='Ruta donde guardar el texto extraído (default: texto_extraido.txt)')
    
    # Argumentos de procesamiento
    parser.add_argument('--sin-recortar',
                        action='store_true',
                        help='No recortar el recibo automáticamente del fondo')
    
    parser.add_argument('--guardar-pasos',
                        action='store_true',
                        help='Guardar imágenes intermedias de cada paso del procesamiento')
    
    parser.add_argument('--sin-ocr',
                        action='store_true',
                        help='Solo mejorar la imagen sin ejecutar OCR')
    
    parser.add_argument('-l', '--idioma',
                        default='spa',
                        help='Idioma para OCR (spa=español, eng=inglés, default: spa)')
    
    # Argumentos avanzados - Parámetros de procesamiento
    parser.add_argument('-r', '--resolucion',
                        type=int,
                        default=1000,
                        help='Altura mínima en píxeles para redimensionar (default: 1000)')
    
    parser.add_argument('-n', '--nivel-ruido',
                        type=int,
                        default=10,
                        choices=range(1, 21),
                        metavar='[1-20]',
                        help='Nivel de eliminación de ruido (1-20, default: 10)')
    
    parser.add_argument('-c', '--contraste',
                        type=float,
                        default=2.0,
                        help='Factor de mejora de contraste CLAHE (default: 2.0)')
    
    parser.add_argument('-k', '--kernel',
                        type=int,
                        default=11,
                        help='Tamaño del kernel para binarización adaptativa (default: 11)')
    
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s 1.0')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Validar que el archivo de entrada existe
    if not os.path.exists(args.imagen):
        print(f"❌ Error: El archivo '{args.imagen}' no existe")
        sys.exit(1)
    
    # Mostrar configuración
    print("="*60)
    print("CONFIGURACIÓN DE PROCESAMIENTO")
    print("="*60)
    print(f"📁 Imagen de entrada: {args.imagen}")
    print(f"💾 Imagen de salida: {args.salida_imagen}")
    print(f"📄 Texto de salida: {args.salida_texto}")
    print(f"✂️  Recortar fondo: {'No' if args.sin_recortar else 'Sí'}")
    print(f"📸 Guardar pasos: {'Sí' if args.guardar_pasos else 'No'}")
    print(f"🔤 Ejecutar OCR: {'No' if args.sin_ocr else 'Sí (' + args.idioma + ')'}")
    print(f"⚙️  Resolución mínima: {args.resolucion}px")
    print(f"🔧 Nivel de ruido: {args.nivel_ruido}")
    print(f"🌓 Factor contraste: {args.contraste}")
    print(f"📐 Kernel size: {args.kernel}")
    print("="*60 + "\n")
    
    try:
        # Procesar imagen con parámetros personalizados
        imagen_mejorada = mejorar_imagen_recibo(
            args.imagen, 
            guardar_pasos=args.guardar_pasos, 
            recortar=not args.sin_recortar,
            resolucion_min=args.resolucion,
            nivel_ruido=args.nivel_ruido,
            factor_contraste=args.contraste,
            kernel_binarizacion=args.kernel
        )
        
        # Guardar resultado final
        cv2.imwrite(args.salida_imagen, imagen_mejorada)
        print(f"\n✓ Imagen mejorada guardada en: {args.salida_imagen}")
        
        # Ejecutar OCR si no está deshabilitado
        if not args.sin_ocr:
            print(f"\n🔍 Ejecutando OCR en idioma '{args.idioma}'...")
            texto = pytesseract.image_to_string(imagen_mejorada, lang=args.idioma)
            
            print("\n" + "="*60)
            print("TEXTO EXTRAÍDO DEL RECIBO:")
            print("="*60)
            print(texto)
            
            # Guardar texto en archivo
            with open(args.salida_texto, 'w', encoding='utf-8') as f:
                f.write(texto)
            
            print(f"\n✓ Texto guardado en: {args.salida_texto}")
        
        print("\n✅ Procesamiento completado exitosamente\n")
        
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {str(e)}")
        sys.exit(1)