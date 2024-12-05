from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from gp_retouch.image.image import Image
from gp_retouch.image.image_processor import ImageProcessor


@pytest.fixture
def rgb_image() -> np.ndarray:  # noqa: D103
    image_path = Path("tests") / "data" / "link.png"
    with PILImage.open(image_path) as image:
        image_data = np.array(image.convert("RGB"))
    return Image(image_data)


@pytest.fixture
def image_processor():  # noqa: D103
    return ImageProcessor()


@pytest.mark.parametrize("downscale_factor", [0.8, 0.2, 0.01])
def test_downscale(image_processor, rgb_image, downscale_factor):  # noqa: D103
    downscaled_image = image_processor.downscale(rgb_image, downscale_factor)
    for dim in [0, 1]:
        assert downscaled_image.shape[dim] - (rgb_image.shape[dim] * downscale_factor) < 1


def test_convert_to_grayscale(image_processor, rgb_image):  # noqa: D103
    grayscale_image = image_processor.convert_to_grayscale(rgb_image)
    assert grayscale_image.is_grayscale
