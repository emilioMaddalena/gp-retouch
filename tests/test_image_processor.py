from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gp_retouch.image.image_processor import ImageProcessor


@pytest.fixture
def rgb_image() -> np.ndarray:  # noqa: D103
    image_path = Path("tests") / "data" / "link.png"
    with Image.open(image_path) as image:
        image_array = np.array(image.convert("RGB"))
    return image_array


@pytest.fixture
def image_processor():  # noqa: D103
    return ImageProcessor()


def test_convert_to_greyscale(image_processor, rgb_image):  # noqa: D103
    greyscale_image = image_processor.convert_to_grayscale(rgb_image)
    assert len(greyscale_image.shape) == 2


@pytest.mark.parametrize("downscale_factor", [0.8, 0.2, 0.01])
def test_downscale(image_processor, rgb_image, downscale_factor):  # noqa: D103
    original = image_processor.convert_to_grayscale(rgb_image)
    downscaled = image_processor.downscale(original, downscale_factor)
    image_processor.print(downscaled)
    for dim in [0, 1]:
        assert downscaled.shape[dim] - (original.shape[dim] * downscale_factor) < 1
