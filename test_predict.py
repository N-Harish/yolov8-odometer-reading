from ultralytics import YOLO
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import pandas as pd
from glob import glob
import constants
from pathlib import Path
import os
from tqdm import tqdm


def handle_similar_letters_numbers(text: str) -> str:
    """Utility function to handle misclassification of numbers as alphabets

    Args:
        text (str): Predicted OCR text

    Returns:
        str: Fixed OCR text
    """
    similar_chars = {'O': '0', 'I': '7', 'l': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'T': '7'}
    
    # Replace similar letters with their corresponding numbers
    for char, num in similar_chars.items():
        text = text.replace(char, num)
    
    filtered_text = text
    return filtered_text


def ocr(image, processor, model, device):
    """OCR odometer reading

    Args:
        image : cropped odometer reading
        processor : Huggingface OCR processor
        model : Huggingface OCR model.

    Returns:
        str: OCR'd string as output
    """
    try:
        # We can directly perform OCR on cropped images.
        pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except:
        return ""


def detect_odometer(img_pth: str, yolo_model: YOLO, ocr_model: VisionEncoderDecoderModel, ocr_processor: TrOCRProcessor, device: str = "cpu"):    
    # Run inference
    results = yolo_model.predict(img_pth,max_det=1)
    img = cv2.imread(img_pth)
    preds = []
    try:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                print(b.numpy())

                bbox = b.numpy()
                image = Image.fromarray(img)
                imgcp = image.copy()
                cropped = imgcp.crop((bbox[0], bbox[1],bbox[2],bbox[3]))

                # convert to gray scale since it will give better result for OCR
                cropped = cropped.convert('L')

                # append gray scale image 3 times to create RGB black and white image for OCR model
                cropped = Image.merge('RGB', (cropped, cropped, cropped))

                readings = ocr(cropped, processor=ocr_processor, model=ocr_model, device=device)
                # reaadings = handle_similar_letters_numbers(reaadings)
                preds.append(readings)
        if len(preds) > 1:
            pred = ",".join(preds)
        else:
            pred = preds[0]
        return pred
    except:
        return ""


def initialize_models(device: str = "cpu"):
    # Load model
    yolo_model = YOLO(constants.YOLO_MODEl)
    
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    model = VisionEncoderDecoderModel.from_pretrained(
        'microsoft/trocr-small-printed'
    ).to(device)

    return yolo_model, processor, model


if __name__ == "__main__":
    # device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    # load nmodels
    yolo_model, ocr_processor, ocr_model = initialize_models()
    
    allowed_exts = {".png", ".jpg"}
    files = glob(os.path.join(constants.TEST_FOLDER, "*.jpg"))
    files = [i for i in files if Path(i).suffix in allowed_exts]

    img_names = []
    predictions = []

    for img_pth in tqdm(files):
        p = detect_odometer(img_pth, yolo_model, ocr_model, ocr_processor)
        p = handle_similar_letters_numbers(p)
        print(p)

        img_names.append(str(Path(img_pth).stem))
        predictions.append(p)
    
    df = pd.DataFrame({"image": img_names, "odometer_readings": predictions})
    df.to_csv("odometer_readings.csv", index=False)
    print(df)
    # p = detect_odometer(img_name)
    # print(p)
