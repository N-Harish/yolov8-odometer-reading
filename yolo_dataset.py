import json
from glob import glob
from pprint import pprint
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import cv2
import numpy as np
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import splitfolders
import os


def convert_to_yolo(x, y, xmax, ymax, image_width, image_height):
    # Calculate the center of the bounding box
    x_center = (x + xmax) / 2.0
    y_center = (y + ymax) / 2.0
    
    # Normalize the coordinates
    x_normalized = x_center / image_width
    y_normalized = y_center / image_height
    width_normalized = (xmax - x) / image_width
    height_normalized = (ymax - y) / image_height
    
    return x_normalized, y_normalized, width_normalized, height_normalized


def create_yolo_df(folder_glob: str, new_height: int, new_width: int):
    if not os.path.isdir("yolo"):
        os.mkdir("yolo")

    os.makedirs("./yolo/images", exist_ok=True)
    os.makedirs("./yolo/labels", exist_ok=True)

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

            for i in list(di[i].values())[2]:
                if i['region_attributes']['identity'] == 'odometer':
                    odometer_x = i['shape_attributes']['all_points_x']
                    odometer_y = i['shape_attributes']['all_points_y']
      
            try:
                resized_x = int(min(odometer_x) * scale_x)
                resized_y = int(min(odometer_y) * scale_y)
                resized_w = int(max(odometer_x) * scale_x)
                resized_h = int(max(odometer_y) * scale_y)

                x_yolo, y_yolo, width_yolo, height_yolo = convert_to_yolo(resized_x, resized_y, resized_w, resized_h, new_width, new_height)
                
                with open(f"./yolo/labels/{str(Path(img_name).stem)}.txt", "w") as f:
                    f.write(f"0 {x_yolo} {y_yolo} {width_yolo} {height_yolo}")      
                
                img.save(f"./yolo/images/{img_name}")
            except:
                continue


if __name__ == "__main__":
    ls = "./train/**/via_region_data.json"
    new_width = 640
    # new_height = 480

    new_height = 640

    create_yolo_df(ls, new_height, new_width)

    splitfolders.ratio("yolo", output="yolo_train",
    seed=1337, ratio=(.9, .1), group_prefix=None, move=False)

    print(len(os.listdir("./yolo_train/train/images")))
    print(len(os.listdir("./yolo_train/val/images")))