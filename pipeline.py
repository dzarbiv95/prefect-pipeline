import io
import uuid
from time import sleep

import numpy as np
from PIL import Image
from prefect import task, flow


import tensorflow as tf
from prefect.task_runners import ConcurrentTaskRunner

from db_connection import get_connection
from scrap_images import get_url, save_image, extract_image_urls


DETECT_PEOPLE_MODEL = "lib/saved_models/faster_rcnn-inception_resnet_v2"
DETECT_PEOPLE_THRESHOLD = 0.3
DETECT_PEOPLE_CLASS_LABELS = [69]  # 69 is the label for person

DETECT_FACES_MODEL = "lib/saved_models/faster_rcnn-inception_resnet_v2"
DETECT_FACES_THRESHOLD = 0.3
DETECT_FACES_CLASS_LABELS = [502]  # 502 is the label for human faces


URL_FOR_SCRAP = "https://he.wikipedia.org/wiki/%D7%A2%D7%9E%D7%95%D7%93_%D7%A8%D7%90%D7%A9%D7%99"

concurrency_limit = 2
concurrency_pool = []


def limit_concurrency(p_id: str) -> bool:
    if len(concurrency_pool) < concurrency_limit:
        concurrency_pool.append(p_id)
        return True
    return False


@task(retries=3, retry_delay_seconds=10)
def load_model(model_url_path: str):
    return tf.saved_model.load(model_url_path)


@task
def detect_people(model, img_tensor: tf.Tensor) -> list[dict]:
    p_id = uuid.uuid4().hex
    while not limit_concurrency(p_id):
        sleep(1)
    results = model.signatures['serving_default'](img_tensor)
    concurrency_pool.remove(p_id)
    result = {key: value.numpy() for key, value in results.items()}
    prediction = []
    for i in range(len(result['detection_boxes'])):
            if result['detection_scores'][i] >= DETECT_PEOPLE_THRESHOLD and result['detection_class_labels'][
                    i] in DETECT_PEOPLE_CLASS_LABELS:
                prediction.append({
                    "score": float(result['detection_scores'][i]),
                    "box": [float(v) for v in result['detection_boxes'][i]],
                    "id": uuid.uuid4().hex
                })
    return prediction


@task
def detect_faces(model, img_tensor: tf.Tensor) -> list[dict]:
    results = model.signatures['serving_default'](img_tensor)
    result = {key: value.numpy() for key, value in results.items()}
    prediction = []
    for i in range(len(result['detection_boxes'])):
        if result['detection_scores'][i] >= DETECT_FACES_THRESHOLD and result['detection_class_labels'][
                i] in DETECT_FACES_CLASS_LABELS:
            prediction.append({
                "score": float(result['detection_scores'][i]),
                "box": [float(v) for v in result['detection_boxes'][i]],
                "id": uuid.uuid4().hex
            })
    return prediction


@task
def match_people_and_faces(people_detection: list[dict], face_detection: list[dict]) -> dict:
    match_detections = {}
    for person in people_detection:
        for face in face_detection:
            if (
                    face['box'][0] > person['box'][0] and
                    face['box'][1] > person['box'][1] and
                    face['box'][2] < person['box'][2] and
                    face['box'][3] < person['box'][3]
            ):
                match_detections[person['id']] = face['id']
    return match_detections


@task
def save_results(img_data: dict, match_detections: dict, people_detection: dict, faces_detection: dict) -> None:
    collcetion = get_connection()["images_detection"]
    document = {
        "_id": img_data['img_name'],
        "people_detection": people_detection,
        "faces_detection": faces_detection,
        "match_detections": match_detections
    }
    collcetion.replace_one({"_id": img_data['img_name']}, document, upsert=True)


@task
def process_img(img_data_dict: dict) -> dict:
    img = Image.open(io.BytesIO(img_data_dict['img_data']))
    img = img.resize((512, 512))  # Resize image as required by the model
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0  # Normalize the image
    np_arr = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(np_arr, dtype=tf.float32)


@flow(log_prints=True, task_runner=ConcurrentTaskRunner())
def pipeline():
    web_page = get_url(URL_FOR_SCRAP)
    people_detection_model = load_model.submit(DETECT_PEOPLE_MODEL)
    faces_detection_model = load_model.submit(DETECT_FACES_MODEL)

    img_urls: list = extract_image_urls(web_page.text, URL_FOR_SCRAP)
    for i, img_url in enumerate(img_urls):
        if i > 5:
            break
        pf_response = get_url.submit(img_url)
        img_data_dict = save_image.submit(pf_response)
        img_tf = process_img.submit(img_data_dict)
        people_detection = detect_people.submit(people_detection_model, img_tf)
        faces_detection = detect_faces.submit(faces_detection_model, img_tf)
        match_detections = match_people_and_faces.submit(people_detection, faces_detection)
        save_results.submit(img_data_dict, match_detections, people_detection, faces_detection)


if __name__ == '__main__':
    pipeline()
