import hashlib
import logging
import os

import requests
from urllib import parse
from bs4 import BeautifulSoup


urls_queue = [
    'https://he.wikipedia.org/wiki/%D7%90%D7%93%D7%9D',
]


class WebScraper:

    output_folder: str = None
    scrap_url: str = None
    _scrap_images_generator = None

    def __init__(self, output_folder: str):
        self.scrap_url = self.get_url_for_scrap()
        self.output_folder = output_folder

    @staticmethod
    def get_url_for_scrap() -> str:
        return urls_queue.pop(0)

    @staticmethod
    def calc_abs_url(scrap_url: str, url: str) -> str:
        if url.startswith("http"):
            return url
        if url.startswith("//"):
            return f'http:{url}'
        if url.startswith("/"):
            base_url_parts = parse.urlparse(scrap_url)
            return f'{base_url_parts.scheme}://{base_url_parts.netloc}{url}'
        logging.warning(f"unknown url type: {url}")
        return url

    @staticmethod
    def image_name(img_data: bytes, img_url: str):
        hash_name = hashlib.sha256(img_data).hexdigest()
        return hash_name + "." + img_url.split(".")[-1]

    def fetch_image(self, img_url):
        if not img_url:
            return None
        full_img_url = self.calc_abs_url(self.scrap_url, img_url)
        print("fetch_image:", img_url, full_img_url, self.scrap_url)
        img_data = requests.get(full_img_url).content
        return img_data

    def save_image(self, img_data, file_name):
        if not os.path.exists(f'{self.output_folder}'):
            os.makedirs(f'{self.output_folder}')
        with open(f'{self.output_folder}/{file_name}', 'wb') as f:
            f.write(img_data)
        return file_name

    def fetch_and_save_images(self):
        response = requests.get(self.scrap_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')

        for img in images:
            src = img.get('src')
            if src and src.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
                img_data = self.fetch_image(src)
                if not img_data:
                    continue
                img_name = self.image_name(img_data, src)
                self.save_image(img_data, img_name)
                yield img_data, img_name

        # add more urls to the queue
        urls = soup.find_all('a')
        for url in urls:
            href = url.get('href')
            if href:
                urls_queue.append(self.calc_abs_url(self.scrap_url, href))

    def next_image(self):
        if not self._scrap_images_generator:
            self._scrap_images_generator = self.fetch_and_save_images()
        return next(self._scrap_images_generator)
