import flet as ft
import cv2
import pytesseract
import imutils
import numpy as np

def main(page: ft.Page):
    btn_archivo = ft.ElevatedButton(text="Seleccionar Archivo", on_click=lambda e: btn_archivo_clicked(e))
    txt_archivo = ft.TextField(label="Archivo Seleccionado", width=400)
    btn_procesar = ft.ElevatedButton(text="Procesar Imagen", on_click=lambda e: btn_procesar_clicked(e))
    
    def file_picker_result(e: ft.FilePickerResultEvent):
        if e.files:
            selected_file = e.files[0]
            txt_archivo.value = selected_file.path            
            page.update()
    
    file_picker = ft.FilePicker(on_result=file_picker_result)


    def btn_archivo_clicked(e):
        page.overlay.append(file_picker)
        file_picker.pick_files(allow_multiple=False)
        page.update()

    def btn_procesar_clicked(e):
        imagen = cv2.imread(txt_archivo.value)
        # cambia tamaño de imagen
        #imagen = cv2.resize(imagen, (600, 400))
        #cv2.imshow("Image Window Title", imagen)
        #cv2.waitKey(0)
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("gray.png", gray)
        #cv2.imshow("Image Window Title", gray)
        #cv2.waitKey(0)
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cv2.imwrite("thresh.png", thresh)
        # muestra la imagen umbralizada        
        #cv2.imshow("Image Window Title", thresh[1])
        #cv2.waitKey(0)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist = (dist * 255).astype("uint8")
        #cv2.imshow("Dist", dist)
        #cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)
        cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        chars = []
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # check if contour is at least 35px wide and 100px tall, and if
            # so, consider the contour a digit
            if w >= 35 and h >= 100:
                chars.append(c)

        chars = np.vstack([chars[i] for i in range(0, len(chars))])
        hull = cv2.convexHull(chars)
        # allocate memory for the convex hull mask, draw the convex hull on
        # the image, and then enlarge it via a dilation
        mask = np.zeros(imagen.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [hull], -1, 255, -1)
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow("Mask", mask)
        # take the bitwise of the opening image and the mask to reveal *just*
        # the characters in the image
        final = cv2.bitwise_and(opening, opening, mask=mask)
        # guardar imagen
        cv2.imwrite("final.png", final)

        texto = pytesseract.image_to_string(gray, lang='spa', config='--psm 6')
        print("Texto Detectado:")
        print(texto)


    page.add(
        ft.SafeArea(
            ft.Container(
                content=ft.Column(
                    controls=[
                        btn_archivo,
                        txt_archivo,
                        btn_procesar,
                        file_picker,
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