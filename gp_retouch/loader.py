from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .image.image import Image


def load_image(path_str: str, metadata: dict) -> Image:
    """_summary_.

    Args:
        path_str (str): _description_
        metadata (dict): _description_

    Returns:
        Image: _description_
    """
    image = PILImage.open(path_str)
    image_data = np.array(image.convert("RGB"))
    return Image(data=image_data, metadata=metadata)
