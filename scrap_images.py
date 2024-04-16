import os.path
from urllib import parse
import hashlib


from httpx import Response
from prefect import task
from prefect.concurrency.sync import rate_limit


from bs4 import BeautifulSoup
import httpx

OUTPUT_FOLDER = "saved_images"


@task(retries=3, retry_delay_seconds=30)
def get_url(url: str):
    with httpx.Client(follow_redirects=True) as client:
        res = client.get(url)
        return res


def calc_abs_url(url: str, base_url: str) -> str:
    if url.startswith("http"):
        return url
    if url.startswith("//"):
        return f'http:{url}'
    if url.startswith("/"):
        base_url_parts = parse.urlparse(base_url)
        return f'{base_url_parts.scheme}://{base_url_parts.netloc}{url}'


def extract_image_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    images = soup.find_all('img')
    img_urls = []
    for img in images:
        src = img.get('src')
        if src and src.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
            img_urls.append(calc_abs_url(src, base_url))
    return img_urls


@task
def save_image(response: Response) -> dict:
    img_data = response.content
    img_hash = hashlib.sha256(img_data).hexdigest()
    img_name = img_hash + "." + str(response.url).split(".")[-1].lower()
    with open(os.path.join(OUTPUT_FOLDER, img_name), "wb") as f:
        f.write(img_data)
    return {"img_name": img_name, "img_data": img_data}
