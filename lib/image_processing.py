import io

import numpy as np
from PIL import Image


class ImageProcessing:

    @staticmethod
    def convert_image_to_array(img: bytes or str, size: tuple) -> np.array:
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        img = img.resize(size)  # Resize image as required by the model
        img = img.convert("RGB")
        img_array = np.array(img) / 255.0  # Normalize the image
        return np.expand_dims(img_array, axis=0)
