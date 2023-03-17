import base64

import cv2
import joblib
import json
import numpy as np
import base64
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

__classes = []
__model = None

def classify_image(image_base64_data, file_path=None):
    if file_path:
        img = cv2.imread(file_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    images = preprocess_image(img)
    return predict_image_class(images)

def predict_image_class(images):
    results = __model.predict(images)
    # Get the class predictions
    class_index = np.argmax(results)
    # get the maximum probability
    max_probability = np.round(np.max(results) * 100, 2)
    # if the max similarity is less than 50% return unknown else return class name
    print(max_probability)
    if max_probability < 80.0:
        return "unknown"
    else:
        return __classes[class_index]

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    # Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
    images = np.expand_dims(img_array, axis=0)
    images = preprocess_input(images)
    return images
    # results = model.predict(images)

def load_saved_artifacts():
    print("Loading saved artifacts....")
    # Load the json file that contains the model's structure
    f = Path("artifacts/model_structure.json")
    model_structure = f.read_text()
    # Recreate the Keras model object from the json data
    global __model
    if __model is None:
        __model = model_from_json(model_structure)
        # Re-load the model's trained weights
        __model.load_weights("artifacts/model_weights.h5")
    global __classes
    # load the class names from saved files
    # Open the JSON file
    with open('artifacts/categories.json') as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    # Convert the JSON data to a Python list
    __classes = list(data)
    print("successfully loaded saved artifacts!")
    return

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_b64_test_image_for_gotu():
    with open("b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()
    # classify_image(get_b64_test_image_for_gotu(), None)
    print(classify_image(None, '../lac3.jpg'))