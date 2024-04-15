import os.path

import tensorflow_hub as hub
import tensorflow as tf

MODEL_URL = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
MODELS_DIR = "lib/saved_models"
MODEL_NAME = "faster_rcnn-inception_resnet_v2"

model = hub.load(MODEL_URL)

tf.saved_model.save(model, os.path.join(MODELS_DIR, MODEL_NAME), signatures=model.signatures['default'])