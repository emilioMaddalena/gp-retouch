from pathlib import Path

import numpy as np
import pytest

import gp_retouch
from gp_retouch.image.image import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE, Image
from gp_retouch.image.image_processor import ImageProcessor


@pytest.fixture
def rgb_image() -> np.ndarray:  # noqa: D103
    image_path = Path("tests") / "data" / "link.png"
    return gp_retouch.load_image(image_path)


@pytest.fixture
def grayscale_image() -> np.ndarray:  # noqa: D103
    image_path = Path("tests") / "data" / "pikachu.png"
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


@pytest.mark.parametrize("variance", [0, 10, 100, 1000])
def test_add_gaussian_noise(grayscale_image, variance):  # noqa: D103
    noisy_image = ImageProcessor.add_noise(grayscale_image, method="gaussian", variance=variance)
    assert isinstance(noisy_image, Image)
    assert noisy_image.shape == grayscale_image.shape
    assert noisy_image.get_completeness_ratio() == 1.0
    if variance == 0:
        assert np.all(noisy_image.data == grayscale_image.data)


@pytest.mark.parametrize(
    ("amount", "salt_ratio"),
    [
        (0.1, 0.1),
        (0.9, 0.1),
        (0.1, 0.9),
        (0.9, 0.9),
    ],
)
def test_add_salt_and_pepper_noise(grayscale_image, amount, salt_ratio):  # noqa: D103
    noisy_image = ImageProcessor.add_noise(
        grayscale_image, method="salt_and_pepper", amount=amount, salt_ratio=salt_ratio
    )
    assert isinstance(noisy_image, Image)
    assert noisy_image.shape == grayscale_image.shape
    assert noisy_image.get_completeness_ratio() == 1.0


@pytest.mark.parametrize("variance", [0, 10, 100, 1000])
def test_add_speckle_noise(grayscale_image, variance):  # noqa: D103
    noisy_image = ImageProcessor.add_noise(grayscale_image, method="speckle", variance=variance)
    assert isinstance(noisy_image, Image)
    assert noisy_image.shape == grayscale_image.shape
    assert noisy_image.get_completeness_ratio() == 1.0
    if variance == 0:
        assert np.all(noisy_image.data == grayscale_image.data)


@pytest.mark.parametrize("intensity", [0, 10, 100, 1000])
def test_add_uniform_noise(grayscale_image, intensity):  # noqa: D103
    noisy_image = ImageProcessor.add_noise(grayscale_image, method="uniform", intensity=intensity)
    assert isinstance(noisy_image, Image)
    assert noisy_image.shape == grayscale_image.shape
    assert noisy_image.get_completeness_ratio() == 1.0
    if intensity == 0:
        assert np.all(noisy_image.data == grayscale_image.data)


def test_conform_to_image_data_reqs():  # noqa: D103
    bad_data = np.full((3, 3), MIN_PIXEL_VALUE - 1)
    with pytest.raises(ValueError):
        Image(data=bad_data)  # Should raise a ValueError
    good_data = ImageProcessor._conform_to_image_data_reqs(bad_data)
    try:
        Image(data=good_data)  # Should not raise an error
    except ValueError:
        pytest.fail("ValueError was raised")

    bad_data = np.full((3, 3), MAX_PIXEL_VALUE + 1)
    with pytest.raises(ValueError):
        Image(data=bad_data)  # Should raise a ValueError
    good_data = ImageProcessor._conform_to_image_data_reqs(bad_data)
    try:
        Image(data=good_data)  # Should not raise an error
    except ValueError:
        pytest.fail("ValueError was raised")
