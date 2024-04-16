import os.path
from time import time

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

MODELS_DIR = "../lib/saved_models"
MODEL_NAME = "faster_rcnn-inception_resnet_v2"

IMAGES_FOLDER = "test_images"
t0 = time()
loaded_model = tf.saved_model.load(os.path.join(MODELS_DIR, MODEL_NAME))
print(time() - t0)
model_fn = loaded_model.signatures['serving_default']

img_path = os.path.join(IMAGES_FOLDER, os.listdir(IMAGES_FOLDER)[0])
img = Image.open(img_path)
img_resized = img.resize((512, 512))  # Resize image as required by the model
img_array = np.array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
print(img_array.shape)
img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
print(img_tensor.shape)

results = model_fn(img_tensor)
result = {key: value.numpy() for key, value in results.items()}

image_with_boxes = img_resized.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Loop through each detection and draw the boxes
for i in range(len(result['detection_boxes'])):
    if result['detection_scores'][i] >= 0.3:  # Only consider detections with a confidence score above a threshold
        ymin, xmin, ymax, xmax = result['detection_boxes'][i]
        (left, right, top, bottom) = (xmin * img_resized.width, xmax * img_resized.width,
                                      ymin * img_resized.height, ymax * img_resized.height)
        image_with_boxes = img_resized.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=2)
        plt.figure(figsize=(12, 8))
        plt.title(str(result['detection_class_labels'][i]) + str(result['detection_class_entities'][i]) + str(round(result['detection_scores'][i],2))+ "-" + str(result['detection_boxes'][i]))
        plt.imshow(image_with_boxes)
        plt.savefig(os.path.join(IMAGES_FOLDER, "output", ".".join(img_path.split(".")[:-1] +[ str(i)] + img_path.split(".")[-1:])))
