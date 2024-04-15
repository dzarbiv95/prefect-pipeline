import logging

import numpy as np
from prefect import task, flow
from lib.scrape_images import WebScraper
from lib.image_processing import ImageProcessing
from lib.object_detection import DetectPeople, DetectFaces

IMAGES_FOLDER = 'your-s3-bucket'
LIMIT_SIZE = (1024, 1024)


@task
def scrape_images(scraper: WebScraper):
    next_img_data = scraper.next_image()
    if next_img_data is not None:
        return next_img_data
    return None, None


@task
def process_images(img: bytes):
    return ImageProcessing.convert_image_to_array(img, LIMIT_SIZE)


@task
def detect_people(img_arr: np.array):
    return DetectPeople.detect_people(img_arr)


@task
def detect_faces(img_arr: np.array):
    return DetectFaces.detect_faces(img_arr)


@task
def match_people_faces(people, faces):
    match = {}
    for person in people:
        p_ymin, p_xmin, p_ymax, p_xmax = person['detection_boxes']
        for face in faces:
            f_ymin, f_xmin, f_ymax, f_xmax = face['detection_boxes']
            if p_ymin < f_ymin and p_ymax > f_ymax and p_xmin < f_xmin and p_xmax > f_xmax:
                match[person['id']] = face['id']
                break
    return match


@task
def save_results(img_name, people, faces, match):
    # Save to database
    print(f'Saving results for image {img_name}')
    print("people:", people)
    print("faces:", faces)
    print("match:", match)
    pass


@task(name="process one image", retries=3)
def process_one_image(scraper: WebScraper) -> bool:
    img_data, img_name = scrape_images(scraper)
    if img_name is None:
        return False
    logging.info(f'Processing image {img_name}')
    processed_img = process_images(img_data)
    logging.info(f'Detecting people and faces in image {img_name}')
    people = detect_people(processed_img)
    faces = detect_faces(processed_img)
    match = match_people_faces(people, faces)
    logging.info(f'Saving results for image {img_name}')
    save_results(img_name, people, faces, match)
    return True


@flow(name='ImageProcessing Flow', log_prints=True)
def father_flow():
    scraper = WebScraper(IMAGES_FOLDER)
    exists_images = True
    while exists_images:
        exists_images = process_one_image(scraper)


if __name__ == '__main__':
    father_flow()
