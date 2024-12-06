from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .image.image import Image


def load_image(path_str: str | Path, metadata: dict = {}) -> Image:
    """Load the target image.

    Returns an Image object with the appropriate data in it.
    Additional logic is embedded here to detect grayscale images
    that were saved as RGB are recognized as grayscale.

    Args:
        path_str (str | Path): path to the image.
        metadata (dict): any extra info to be attached to it.

    Returns:
        Image: the final Image object.
    """
    image = PILImage.open(path_str)
    if image.mode == "L":  # grayscale
        image_data = np.array(image)
    elif image.mode in ["RGB", "RGBA"]:
        image_data = np.array(image.convert("RGB"))
        # Check if it was saved as RGB but actually is gray
        if has_identical_channels(image_data):
            image_data = image_data[:, :, 0]
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")
    return Image(data=image_data, metadata=metadata)


def has_identical_channels(image_data: np.ndarray) -> bool:
    """Check if an RGB is faking it.

    Verifies if all three channels of an RGB image has the same
    information. That is, if it's in reality a grayscale image.

    Args:
        image_data (np.ndarray): the image to check.

    Returns:
        bool: if all its channels contain the same array.
    """
    if np.all(image_data[:, :, 0] == image_data[:, :, 1]) & np.all(
        image_data[:, :, 1] == image_data[:, :, 2]
    ):
        return True
    else:
        return False
