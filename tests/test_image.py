import numpy as np

from gp_retouch.image.image import Image


def test_instantiation():  # noqa: D103
    data = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    metadata = {"name": "my random grayscale image", "year": 1993}
    image = Image(data, metadata)

    assert np.all(image.data == data), "Image modified the input data during instantiation"
    assert image.metadata == metadata, "Image modified metadata during instantiation"


def test_is_grayscale():  # noqa: D103
    grayscale_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    image = Image(grayscale_image)
    assert image.is_grayscale, "Image did not recognize a grayscale image"

    rgb_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    image = Image(rgb_image)
    assert not image.is_grayscale, "Image recognized an rgb image as grayscale"


def test_is_rgb():  # noqa: D103
    grayscale_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    image = Image(grayscale_image)
    assert not image.is_rgb, "Image recognized a grayscale image as rgb"

    rgb_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    image = Image(rgb_image)
    assert image.is_rgb, "Image failed to recognize an rgb image"
