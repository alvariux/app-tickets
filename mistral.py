from mistralai import Mistral
import os
import argparse
import re
import base64
import json

ap = argparse.ArgumentParser()
ap.add_argument("image_path", type=str, help="Path to the image file for OCR")
args = vars(ap.parse_args())

with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    with open(args["image_path"], "rb") as image_file:
        image_data = image_file.read()
        # base64 encode the image data as url string        
        encoded_image = f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"

    response = mistral.ocr.process(
        model="mistral-ocr-latest",
        document={
            "image_url": {
                "url": encoded_image,
            },
            "type": "image_url"
        },        
    )

markdown_text = response.pages[0].markdown

estacion = re.search(r"Estacion:\s*(\w+)", markdown_text)
folio = re.search(r"Folio:\s*([\d/-]+)", markdown_text)
webid = re.search(r"WebID:\s*(\w+)", markdown_text)
fecha = re.search(r"\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", markdown_text)
hora = re.search(r"\s*\b(([01]?[0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]))\b", markdown_text)
total = re.search(r"TOTAL:\s*\$?([\d,]+\.\d{2})", markdown_text)

result = {
    "Estacion": estacion.group(1) if estacion else None,
    "Folio": folio.group(1) if folio else None,
    "WebID": webid.group(1) if webid else None,
    "Fecha": fecha.group(1) if fecha else None,
    "Hora": hora.group(1) if hora else None,
    "Total": total.group(1) if total else None,
}

print(json.dumps(result, ensure_ascii=False, indent=2))
