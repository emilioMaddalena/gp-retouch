from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

import gp_retouch
from gp_retouch.image.image import Image
from gp_retouch.image.image_processor import ImageProcessor


@pytest.fixture
def rgb_image() -> np.ndarray:  # noqa: D103
    image_path = Path("tests") / "data" / "link.png"
    return gp_retouch.load_image(image_path)


@pytest.mark.parametrize("downscale_factor", [0.8, 0.2, 0.01])
def test_downscale(rgb_image, downscale_factor):  # noqa: D103
    downscaled_image = ImageProcessor.downscale(rgb_image, downscale_factor)
    for dim in [0, 1]:
        assert downscaled_image.shape[dim] - (rgb_image.shape[dim] * downscale_factor) < 1


def test_convert_to_grayscale(rgb_image):  # noqa: D103
    grayscale_image = ImageProcessor.convert_to_grayscale(rgb_image)
    assert grayscale_image.is_grayscale


@pytest.mark.parametrize("ratio", [0.1, 0.5, 0.8])
def test_drop_pixels(rgb_image, ratio):  # noqa: D103
    num_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    new_image = ImageProcessor.drop_pixels(rgb_image, ratio)
    num_pixels_new = new_image.shape[0] * new_image.shape[1]
    assert (num_pixels * ratio) - num_pixels_new < 1
