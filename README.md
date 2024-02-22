# Explaining Approaches and Procedure to Run Code


## To run the code:

1. Change the "TEST_FOLDER" variable in the "constants.py" file to the path of your test file.
2. Run the "test_predict.py" file. It will generate an "odometer_readings.csv" file containing the image and prediction columns.


## Steps Taken:

1. Converting data for YoloV8 object detection to detect odometer readings:

    - Initially, the JSON annotation paths from sub-directories were stored in a list using `glob`.
    - These JSON files were then read and iterated over.
    - Images without odometer readings were removed, and those with odometer readings were read.
    - Both the bounding boxes and the images were resized to 640 x 640 dimensions.
    - The scaled bounding boxes were converted to YOLO coordinates using the "convert_to_yolo" function in the "yolo_dataset.py" file.
    - Since we only had one class, the coordinates were stored in "./yolo/labels" folder as "x_normalized, y_normalized, width_normalized, height_normalized".
    - The images were stored in the "./yolo/images" folder.


2. Splitting Data into train and validation sets:

    - The `splitfolders` package was used to split the data into train and validation sets.
    - 90% of the data was allocated for training, and the remainder was used for the validation set.
    - These sets were created in separate folders under "yolo_train".


3. Training YoloV8:

    - Once the data was split, a "data.yaml" file was created.
    - Inside "data.yaml", configurations for the YOLO model were specified:
        - "train": The path of the training dataset.
        - "val": The path of the validation dataset.
        - "nc" (number_of_classes) was set to 1 since only the odometer needed to be detected.
        - The "names" (label name) parameter was set to "odometer".
    - Following this, the YoloV8 model was trained.
    - The bash script "train_yolo.sh" was used to train the model for 40 epochs.


4. Complete Pipeline for OCRing odometer reading:

    - First, YOLO is used to detect the odometer reading.
    - The detected odometer is cropped out and converted to grayscale.
    - Grayscale conversion ensures that the LCD lighting doesn't affect the OCR.
    - The grayscale image is then copied thrice to generate a black-and-white RGB image (important since the OCR model works only on RGB images).
    - This image is passed to the TrOCR model, a transformer-based OCR model created by Microsoft Research.
    - After receiving a prediction, it is passed to the "handle_similar_letters_numbers" function, 
      ensuring that alphabet characters similar to numbers are converted back to numbers
      (e.g., {'O': '0', 'I': '1', 'l': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'T': '7'}).