import uuid

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class ObjectDetection:
    _model: tf = None
    _model_name: str = None
    _threshold: float = 0.3
    _detection_class_labels: list = None

    def __init__(self, model_path: str, detection_class_labels: list = None, accuracy_threshold: float = 0.3):
        self._model_name = model_path
        self._threshold = accuracy_threshold
        self._detection_class_labels = detection_class_labels.copy()
        self._model = hub.load(model_path)

    def predict(self, img_tensor: tf.Tensor or np.array):
        """
        Predict the image

        :param img_tensor: tf.Tensor: Image tensor to predict.
        the dimension of the image should be (1, height, width, 3)

        :return: list: List of predictions
        """
        if isinstance(img_tensor, np.ndarray):
            img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
        results = self._model.signatures['default'](img_tensor)
        result = {key: value.numpy() for key, value in results.items()}
        prediction = []
        for i in range(len(result['detection_boxes'])):
            if result['detection_scores'][i] >= self._threshold and result['detection_class_labels'][
                    i] in self._detection_class_labels:
                prediction.append({
                    "label": result['detection_class_labels'][i],
                    "entity": result['detection_class_entities'][i],
                    "score": result['detection_scores'][i],
                    "box": result['detection_boxes'][i],
                    "id": uuid.uuid4().hex
                })
        return prediction


class DetectPeople:

    object_detection_model: ObjectDetection = None
    _initialized = False

    @classmethod
    def init(cls):
        cls.object_detection_model = ObjectDetection(
            model_path="google/faster-rcnn-inception-resnet-v2/tensorFlow1/faster-rcnn-openimages-v4-inception-resnet-v2",
            detection_class_labels=[69],  # 69 is the label for person
            accuracy_threshold=0.3
        )
        cls._initialized = True

    @classmethod
    def detect_people(cls, img_tensor: tf.Tensor or np.array):
        if not cls._initialized:
            cls.init()
        return cls.object_detection_model.predict(img_tensor)


class DetectFaces:

    object_detection_model: ObjectDetection = None
    _initialized = False

    @classmethod
    def init(cls):
        cls.object_detection_model = ObjectDetection(
            model_path="google/faster-rcnn-inception-resnet-v2/tensorFlow1/faster-rcnn-openimages-v4-inception-resnet-v2",
            detection_class_labels=[502],  # 502 is the label for human faces
            accuracy_threshold=0.3
        )
        cls._initialized = True

    @classmethod
    def detect_faces(cls, img_tensor: tf.Tensor or np.array):
        if not cls._initialized:
            cls.init()
        return cls.object_detection_model.predict(img_tensor)