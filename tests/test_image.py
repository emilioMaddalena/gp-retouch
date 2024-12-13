import numpy as np
import pytest

from gp_retouch.image.image import Image


@pytest.fixture
def grayscale_image_data():  # noqa: D103
    return np.random.randint(0, 256, (50, 50), dtype=np.uint8)


@pytest.fixture
def rgb_image_data():  # noqa: D103
    return np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_nonsquare_data():  # noqa: D103
    return np.random.randint(0, 256, (30, 50), dtype=np.uint8)


@pytest.fixture
def image_data_with_nans():  # noqa: D103
    # Generate random RGB image
    data = np.random.random((50, 50, 3))
    # Randomly decide indices to drop
    num_nans = np.random.randint(1, 50)
    nan_indices = (
        np.random.choice(50, size=num_nans, replace=False),
        np.random.choice(50, size=num_nans, replace=False),
    )
    # Drop uniformily from all 3 channels
    for i, j in zip(*nan_indices):
        data[i, j, :] = np.nan  # Drop entries in all channels
    return data


def test_instantiation(grayscale_image_data):  # noqa: D103
    metadata = {"name": "my random grayscale image", "year": 1993}
    image = Image(grayscale_image_data, metadata)

    assert np.all(
        image.data == grayscale_image_data
    ), "Image modified the input data during instantiation"
    assert image.metadata == metadata, "Image modified metadata during instantiation"


def test_is_grayscale(grayscale_image_data, rgb_image_data):  # noqa: D103
    image = Image(grayscale_image_data)
    assert image.is_grayscale, "Image did not recognize a grayscale image"

    image = Image(rgb_image_data)
    assert not image.is_grayscale, "Image recognized an rgb image as grayscale"


def test_is_rgb(grayscale_image_data, rgb_image_data):  # noqa: D103
    image = Image(grayscale_image_data)
    assert not image.is_rgb, "Image recognized a grayscale image as rgb"

    image = Image(rgb_image_data)
    assert image.is_rgb, "Image failed to recognize an rgb image"


def test_height_width(grayscale_nonsquare_data):  # noqa: D103
    height, width = grayscale_nonsquare_data.shape
    image = Image(grayscale_nonsquare_data)
    assert image.height == height
    assert image.width == width


def test_is_incompelte(image_data_with_nans, grayscale_image_data):  # noqa: D103
    image = Image(image_data_with_nans)
    assert image.is_incomplete, "Image failed to recognize incomplete data"

    image = Image(grayscale_image_data)
    assert not image.is_incomplete, "Image recognized incomplete data when data was complete"


def test_get_completeness_ratio(grayscale_image_data):  # noqa: D103
    # Test complete image
    image = Image(grayscale_image_data)
    assert (
        image.get_completeness_ratio() == 1.0
    ), "Image recognized incomplete data when data was complete"

    # Incomplete image
    nan_ratio = 0.2

    data_shape = (50, 50, 3)
    data = np.random.rand(*data_shape)
    total_elements = np.prod(data_shape)
    num_nans = int(total_elements * nan_ratio)
    flat_data = data.flatten()
    nan_indices = np.random.choice(len(flat_data), num_nans, replace=False)
    flat_data[nan_indices] = np.nan
    data = flat_data.reshape(data_shape)

    image = Image(data)
    np.testing.assert_approx_equal(
        image.get_completeness_ratio(),
        1 - nan_ratio,
        significant=3,
        err_msg="Image did not recognize the right completness ratio",
    )
