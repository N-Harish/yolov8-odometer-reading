import json
from glob import glob
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import random
import shutil


def create_ocr_data(folder_glob, new_height = 640, new_width = 640):
    ls = glob(folder_glob)

    for pth in tqdm(sorted(ls), total=len(ls)):

        with open(pth, "r") as f:
            di = json.load(f)

        img_folder = str(Path(pth).parent)

        for i in di.keys():
            img_name = list(di[i].values())[0]
            img_name = str(Path(img_name).parts[-1])
            
            img_pth = os.path.join(img_folder, img_name)
            img = Image.open(img_pth)

            w, h = img.size

            img = img.resize((new_width, new_height))

            scale_x = new_width / w
            scale_y = new_height / h

            odometer_x = []
            odometer_y = []
            reading = None
            for i in list(di[i].values())[2]:
                if i['region_attributes']['identity'] == 'odometer':
                    odometer_x = i['shape_attributes']['all_points_x']
                    odometer_y = i['shape_attributes']['all_points_y']
                    reading = i['region_attributes'].get('reading', "")

            
            try:
                resized_x = int(min(odometer_x) * scale_x)
                resized_y = int(min(odometer_y) * scale_y)
                resized_w = int(max(odometer_x) * scale_x)
                resized_h = int(max(odometer_y) * scale_y)

                imgcp = img.copy()
                imgcp_draw = ImageDraw.Draw(imgcp)

                
                cropped = imgcp.crop((resized_x, resized_y, resized_w, resized_h))

                if reading:
                    with open(f"./ocr_data/labels.txt", "a") as f:
                        f.write(f"{img_name} {reading}\n")     

                    cropped.save(f"./ocr_data/{img_name}")
            except:
                continue

def remove_empty_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip()]

    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(non_empty_lines)


def split_text_file(input_file, train_file, test_file, split_ratio=0.8):
    # Read the content of the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Shuffle the lines to randomize the order
    random.shuffle(lines)

    # Determine the split index
    split_index = int(len(lines) * split_ratio)

    # Split the lines into train and test sets
    train_data = lines[:split_index]
    test_data = lines[split_index:]

    # Write train data to file
    with open(train_file, 'w', encoding='utf-8') as file:
        file.writelines(train_data)

    # Write test data to file
    with open(test_file, 'w', encoding='utf-8') as file:
        file.writelines(test_data)


def copy_images(label_pth: str, images_fol: str = "./ocr_data"):
  with open(label_pth, 'r') as f:
    li = f.readlines()
  dest_fol =  str(Path(label_pth).parent)
  for lines in tqdm(li):
    # print(lines)
    img_name = lines.split(" ")[0]
    img_pth = os.path.join(images_fol, img_name)
    shutil.copy(img_pth, dest_fol)


if __name__ == "__main__":
    
    if not os.path.isdir("ocr_data"):
        os.mkdir("ocr_data")

    folder_glob = "./train/**/via_region_data.json"

    create_ocr_data(folder_glob)
    remove_empty_lines("./ocr_data/labels.txt", "./ocr_data/labels.txt")

    # split into train and val
    base_pth = "./ocr_data"
    input_file = os.path.join(base_pth,'labels.txt')

    if not os.path.isdir("./train_ocr"):
        os.mkdir("train_ocr")

    if not os.path.isdir("./val_ocr"):
        os.mkdir("val_ocr")

    train_file = './train_ocr/labels.txt'
    test_file = './val_ocr/labels.txt'
    split_ratio = 0.9

    split_text_file(input_file, train_file, test_file, split_ratio)
    copy_images(train_file)
    copy_images(test_file)
