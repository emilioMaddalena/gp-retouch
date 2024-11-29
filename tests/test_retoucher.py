from pathlib import Path

import cv2
import numpy as np
import pytest

from gp_retouch.retoucher import Retoucher


# Helper function to create a mock image for testing
@pytest.fixture
def mock_image(tmp_path):
    # Create a 100x100 grayscale image
    img = np.full((100, 100), 128, dtype=np.uint8)
    file_path = tmp_path / "test_image.png"
    cv2.imwrite(str(file_path), img)
    return file_path


@pytest.fixture
def mock_image_with_nans(tmp_path):
    # Create a 100x100 grayscale image with NaNs
    img = np.full((100, 100), 128, dtype=np.float32)
    img[50, 50] = np.nan  # Add a NaN value
    file_path = tmp_path / "test_image_with_nans.png"
    cv2.imwrite(str(file_path), img)
    return file_path


@pytest.fixture
def retoucher_instance():
    return Retoucher()


def test_load_image_grayscale(mock_image, retoucher_instance):
    retoucher_instance.load_image(mock_image, grayscale=True)
    assert retoucher_instance.image is not None
    assert retoucher_instance.image.shape == (100, 100)


def test_load_image_color(mock_image, retoucher_instance):
    retoucher_instance.load_image(mock_image, grayscale=False)
    assert retoucher_instance.image is not None
    assert len(retoucher_instance.image.shape) == 3  # Color image should have 3 channels


def test_load_image_invalid_path(retoucher_instance):
    with pytest.raises(ValueError):
        retoucher_instance.load_image("invalid_path.jpg")


def test_downscale_image(mock_image, retoucher_instance):
    retoucher_instance.load_image(mock_image, grayscale=True)
    retoucher_instance.downscale_image(0.5)
    assert retoucher_instance.downscaled_image is not None
    assert retoucher_instance.downscaled_image.shape == (50, 50)


def test_downscale_image_without_loading(retoucher_instance):
    with pytest.raises(ValueError):
        retoucher_instance.downscale_image(0.5)


def test_fill_nans(mock_image_with_nans, retoucher_instance):
    def mock_model(image, **kwargs):
        # Replace NaNs with a constant value (e.g., 255)
        image[np.isnan(image)] = 255
        return image

    retoucher_instance.load_image(mock_image_with_nans, grayscale=True)
    retoucher_instance.downscale_image(1.0)  # Keep the same size
    filled_image = retoucher_instance.fill_nans(mock_model)
    assert filled_image is not None
    assert not np.isnan(filled_image).any()


def test_fill_nans_no_downscale(mock_image_with_nans, retoucher_instance):
    def mock_model(image, **kwargs):
        image[np.isnan(image)] = 255
        return image

    retoucher_instance.load_image(mock_image_with_nans, grayscale=True)
    with pytest.raises(ValueError):
        retoucher_instance.fill_nans(mock_model)


def test_visualize_images(mock_image, retoucher_instance):
    retoucher_instance.load_image(mock_image, grayscale=True)
    retoucher_instance.downscale_image(0.5)
    # This is a visual method; we cannot assert much, but ensure it runs without errors
    retoucher_instance.visualize_images()
