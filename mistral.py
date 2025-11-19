from mistralai import Mistral
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("image_path", type=str, help="Path to the image file for OCR")
args = vars(ap.parse_args())


#with Mistral(
#    api_key=os.getenv("MISTRAL_API_KEY", "PbDSfdg5DRmxQ4K7ixdf2DgDJcipVhuh"),
#) as mistral:

#    res = mistral.models.list()

    # Handle response
#    print(res)
    # Print each model on its own line
    
#    for model in res.data:
#        print(f"ID: {model.id}, Name: {model.name}")
#        for capability in model.capabilities:
#            print(f"  - Capability: {capability}")
    
#mistral-ocr-latest

with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    with open(args["image_path"], "rb") as image_file:
        image_data = image_file.read()
        # base64 encode the image data as url string
        import base64
        encoded_image = f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"

    response = mistral.ocr.process(
        model="mistral-ocr-latest",
        document={
            "image_url": {
                "url": encoded_image,
            },
            "type": "image_url"
    })

    print("OCR Result:", response)


