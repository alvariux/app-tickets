import flet as ft
import cv2
import pytesseract
import imutils
import numpy as np
import base64

def main(page: ft.Page):
    btn_archivo = ft.ElevatedButton(text="Seleccionar Archivo", on_click=lambda e: btn_archivo_clicked(e))
    txt_archivo = ft.TextField(label="Archivo Seleccionado", width=400)
    btn_procesar = ft.ElevatedButton(text="Procesar Imagen", on_click=lambda e: btn_procesar_clicked(e))
    img_display = ft.Image(src="blank.png",width=600, height=400,fit=ft.ImageFit.CONTAIN)
    
    def file_picker_result(e: ft.FilePickerResultEvent):
        if e.files:
            selected_file = e.files[0]
            txt_archivo.value = selected_file.path
            img_display.src = selected_file.path
            page.update()
    
    file_picker = ft.FilePicker(on_result=file_picker_result)


    def btn_archivo_clicked(e):
        page.overlay.append(file_picker)
        file_picker.pick_files(allow_multiple=False)
        page.update()

    # Selección de 4 esquinas con clicks sobre la imagen y recorte por perspectiva
    txt_x = ft.TextField(label="P1 (x,y)", width=120, value="")
    txt_y = ft.TextField(label="P2 (x,y)", width=120, value="")
    txt_w = ft.TextField(label="P3 (x,y)", width=120, value="")
    txt_h = ft.TextField(label="P4 (x,y)", width=120, value="")

    puntos = []  # lista de (x,y) en coordenadas de la imagen original

    def update_point_fields():
        vals = [("", txt_x), ("", txt_y), ("", txt_w), ("", txt_h)]
        for i in range(min(4, len(puntos))):
            vals[i] = (f"{puntos[i][0]},{puntos[i][1]}", [txt_x, txt_y, txt_w, txt_h][i])
        # asignar valores
        if len(puntos) > 0:
            txt_x.value = f"{puntos[0][0]},{puntos[0][1]}"
        else:
            txt_x.value = ""
        if len(puntos) > 1:
            txt_y.value = f"{puntos[1][0]},{puntos[1][1]}"
        else:
            txt_y.value = ""
        if len(puntos) > 2:
            txt_w.value = f"{puntos[2][0]},{puntos[2][1]}"
        else:
            txt_w.value = ""
        if len(puntos) > 3:
            txt_h.value = f"{puntos[3][0]},{puntos[3][1]}"
        else:
            txt_h.value = ""
        page.update()

    def reset_points(e=None):
        puntos.clear()
        update_point_fields()

    def order_points(pts):
        # pts: array [[x,y],...]
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def img_tap(e):
        path = txt_archivo.value
        if not path:
            print("Selecciona primero un archivo.")
            return
        imagen = cv2.imread(path)
        if imagen is None:
            print("No se puede leer la imagen.")
            return
        h_img, w_img = imagen.shape[:2]
        # tamaño mostrado del widget
        disp_w = img_display.width or w_img
        disp_h = img_display.height or h_img
        # cálculo escala y offsets para ImageFit.CONTAIN
        scale = min(disp_w / w_img, disp_h / h_img)
        img_disp_w = w_img * scale
        img_disp_h = h_img * scale
        offset_x = (disp_w - img_disp_w) / 2
        offset_y = (disp_h - img_disp_h) / 2

        lx = e.local_x
        ly = e.local_y
        # comprobar si el click está dentro del área de la imagen mostrada
        if lx < offset_x or lx > offset_x + img_disp_w or ly < offset_y or ly > offset_y + img_disp_h:
            print("Click fuera del área de la imagen.")
            return
        # mapear a coordenadas de la imagen original
        x_img = int((lx - offset_x) / scale)
        y_img = int((ly - offset_y) / scale)
        if len(puntos) < 4:
            puntos.append((x_img, y_img))
            print(f"Punto {len(puntos)} = ({x_img},{y_img})")
        else:
            print("Ya hay 4 puntos. Reinicia si quieres seleccionar de nuevo.")
        update_point_fields()

    # envolver la imagen en un GestureDetector para captar clicks
    img_flet = ft.GestureDetector(content=img_display, on_tap_down=lambda e: img_tap(e))

    def btn_recortar_clicked(e):
        if len(puntos) < 4:
            print("Selecciona las 4 esquinas antes de recortar.")
            return
        path = txt_archivo.value
        imagen = cv2.imread(path)
        if imagen is None:
            print("No se pudo leer la imagen para recorte.")
            return
        pts_np = np.array(puntos, dtype="float32")
        rect = order_points(pts_np)
        (tl, tr, br, bl) = rect
        # ancho y alto del rectángulo destino
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(imagen, M, (maxWidth, maxHeight))
        output_path = "cropped.png"
        cv2.imwrite(output_path, warped)
        # actualizar imagen mostrada y ruta para procesos posteriores
        img_flet.content.src = output_path  # img_flet es GestureDetector, su content es la Image
        txt_archivo.value = output_path
        page.update()
        print("Recorte guardado en", output_path)

    # botón principal y reset agrupados (btn_recortar es usado en la fila del layout)
    btn_recortar_main = ft.ElevatedButton(text="Recortar Imagen", on_click=lambda e: btn_recortar_clicked(e))
    btn_reset = ft.ElevatedButton(text="Reset Puntos", on_click=lambda e: reset_points(e))
    btn_recortar = ft.Row(controls=[btn_recortar_main, btn_reset], spacing=10)

    def btn_procesar_clicked(e):
        ruta_imagen = txt_archivo.value
        if ruta_imagen is None:
            print("No se pudo leer la imagen para procesar.")
            return
        
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

        # Paso 4: Aplicar filtro bilateral para reducir ruido manteniendo bordes
        filtrado = cv2.bilateralFilter(contraste, 9, 75, 75)

        # Usar filtro bilateral para preservar bordes
        suavizado = cv2.bilateralFilter(gris, 15, 80, 80)    

        # 4. Umbralización adaptativa MÁS CONSERVADORA
        umbral = cv2.adaptiveThreshold(
            suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 8  # Parámetros más conservadores
        )

        # Solo limpieza mínima
        kernel = np.ones((1, 1), np.uint8)
        procesada = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)

        # Guardar imagen procesada temporalmente
        output_path = "processed.png"
        cv2.imwrite(output_path, procesada)
        img_flet.content.src_base64 = base64.b64encode(cv2.imencode('.png', procesada)[1]).decode('utf-8')                 
        page.update()
        print("Imagen procesada guardada en", output_path)


    def rotar_izquierda(e):
        path = txt_archivo.value
        if not path:
            print("Selecciona primero un archivo.")
            return
        img = cv2.imread(path)
        if img is None:
            print("No se puede leer la imagen para rotar.")
            return
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        output_path = "rotated.png"
        cv2.imwrite(output_path, rotated)
        # actualizar display y estado          
        img_flet.content.src_base64 = base64.b64encode(cv2.imencode('.png', rotated)[1]).decode('utf-8')         
        txt_archivo.value = output_path
        puntos.clear()
        update_point_fields()
        page.update()
        print("Imagen rotada 90° a la izquierda y guardada en", path)

    def rotar_derecha(e):
        path = txt_archivo.value
        if not path:
            print("Selecciona primero un archivo.")
            return
        img = cv2.imread(path)
        if img is None:
            print("No se puede leer la imagen para rotar.")
            return
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        output_path = "rotated.png"
        cv2.imwrite(output_path, rotated)
        # actualizar display y estado        
        img_flet.content.src_base64 = base64.b64encode(cv2.imencode('.png', rotated)[1]).decode('utf-8')         
        txt_archivo.value = output_path
        puntos.clear()
        update_point_fields()
        page.update()
        print("Imagen rotada 90° a la derecha y guardada en", path)

    btn_rot_left = ft.ElevatedButton(text="Rotar 90° Izquierda", on_click=lambda e: rotar_izquierda(e))
    btn_rot_right = ft.ElevatedButton(text="Rotar 90° Derecha", on_click=lambda e: rotar_derecha(e))

    # agregar los botones al Row existente que contiene los botones de recorte/reset
    btn_recortar.controls.append(btn_rot_left)
    btn_recortar.controls.append(btn_rot_right)


    page.add(
        ft.SafeArea(
            ft.Container(
                content=ft.Column(
                    controls=[
                        btn_archivo,
                        txt_archivo,
                        btn_procesar,
                        file_picker,
                        img_flet,
                        ft.Row(
                            controls=[
                                txt_x,
                                txt_y,
                                txt_w,
                                txt_h,
                                btn_recortar
                            ],
                            alignment=ft.MainAxisAlignment.START
                        )
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                ),
                alignment=ft.alignment.center,
            ),
            expand=True,
        )
    )

ft.app(main)