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
from constants import DETECT_PEOPLE_THRESHOLD, DETECT_PEOPLE_CLASS_LABELS, DETECT_PEOPLE_MODEL, \
    DETECT_FACES_THRESHOLD, DETECT_FACES_CLASS_LABELS, DETECT_FACES_MODEL, IMAGE_MAX_SIZE, URL_FOR_SCRAP

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
def process_img(img_data_dict: dict) -> dict:
    img = Image.open(io.BytesIO(img_data_dict['img_data']))
    img = img.resize(IMAGE_MAX_SIZE)  # Resize image as required by the model
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0  # Normalize the image
    np_arr = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(np_arr, dtype=tf.float32)


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
