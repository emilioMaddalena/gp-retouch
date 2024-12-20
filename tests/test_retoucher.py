import GPy
import numpy as np
import pytest

from gp_retouch.image.image import Image
from gp_retouch.retoucher import Retoucher


@pytest.fixture
def grayscale_image():
    """Create a mock grayscale image with missing pixels."""
    data = np.random.rand(10, 10)
    data[2:4, 2:4] = np.nan
    return Image(data=data)


@pytest.fixture
def full_grayscale_image():
    """Create a mock grayscale image."""
    data = np.random.rand(10, 10)
    return Image(data=data)


@pytest.fixture
def rgb_image():
    """Create a mock RGB image with missing pixels."""
    data = np.random.rand(10, 10, 3)
    data[2:4, 2:4, :] = np.nan
    return Image(data=data)


@pytest.fixture
def full_rgb_image():
    """Create a mock RGB image."""
    data = np.random.rand(10, 10, 3)
    return Image(data=data)


def test_load_image(grayscale_image):
    """Test loading an image."""
    retoucher = Retoucher()
    retoucher.load_image(grayscale_image)
    assert retoucher.image is not None
    assert np.allclose(retoucher.image.data, grayscale_image.data, atol=1e-4, equal_nan=True)


def test_learn_image_grayscale(grayscale_image):
    """Test learning the kernel hyperparameters for a grayscale image."""
    retoucher = Retoucher()
    retoucher.load_image(grayscale_image)
    retoucher.learn_image(max_iters=1)
    assert retoucher.gp is not None
    assert isinstance(retoucher.gp, GPy.models.GPRegression)


def test_learn_image_rgb(rgb_image):
    """Test learning the kernel hyperparameters for an RGB image."""
    retoucher = Retoucher()
    retoucher.load_image(rgb_image)
    retoucher.learn_image(max_iters=1)
    assert retoucher.gp is not None
    assert len(retoucher.gp) == 3
    assert all(isinstance(gp_channel, GPy.models.GPRegression) for gp_channel in retoucher.gp)


def test_reconstruct_image_grayscale(grayscale_image):
    """Test reconstructing a grayscale image."""
    retoucher = Retoucher()
    retoucher.load_image(grayscale_image)
    retoucher.learn_image(max_iters=1)
    reconstructed_image = retoucher.reconstruct_image()

    assert isinstance(reconstructed_image, Image)
    assert reconstructed_image.shape == grayscale_image.shape
    assert not np.any(np.isnan(reconstructed_image.data))


def test_denoise_image(full_grayscale_image, full_rgb_image):
    """The denoising quality is subjective and hard to check.

    Here we're just checking for the returned class, shape and 
    the completness ratio.
    """
    # Denoise grayscale
    retoucher = Retoucher()
    retoucher.load_image(full_grayscale_image)
    retoucher.learn_image(max_iters=1)
    denoised_image = retoucher.denoise_image(image=full_grayscale_image, factor=0.5)

    assert isinstance(denoised_image, Image)
    assert denoised_image.shape == full_grayscale_image.shape
    assert denoised_image.get_completeness_ratio() == 1.

    # Denoise rgb
    retoucher = Retoucher()
    retoucher.load_image(full_rgb_image)
    retoucher.learn_image(max_iters=1)
    denoised_image = retoucher.denoise_image(image=full_rgb_image, factor=0.5)

    assert isinstance(denoised_image, Image)
    assert denoised_image.shape == full_rgb_image.shape
    assert denoised_image.get_completeness_ratio() == 1.

    # Denoise factor = 0 -> must return the same input image w/o changes
    retoucher = Retoucher()
    retoucher.load_image(full_grayscale_image)
    retoucher.learn_image(max_iters=1)
    denoised_image = retoucher.denoise_image(image=full_grayscale_image, factor=0)

    assert isinstance(denoised_image, Image)
    assert np.all(denoised_image.data == full_grayscale_image.data)


def test_reconstruct_image_rgb(rgb_image):
    """Test reconstructing an RGB image."""
    retoucher = Retoucher()
    retoucher.load_image(rgb_image)
    retoucher.learn_image(max_iters=1)
    reconstructed_image = retoucher.reconstruct_image()

    assert isinstance(reconstructed_image, Image)
    assert reconstructed_image.shape == rgb_image.shape
    assert not np.any(np.isnan(reconstructed_image.data))


def test_no_image_loaded():
    """Test methods when no image is loaded."""
    retoucher = Retoucher()

    with pytest.raises(ValueError, match="No image loaded"):
        retoucher.learn_image()

    with pytest.raises(ValueError, match="No image loaded"):
        retoucher.reconstruct_image()


def test_no_gp_trained(grayscale_image):
    """Test reconstructing an image before training the GP model."""
    retoucher = Retoucher()
    retoucher.load_image(grayscale_image)

    with pytest.raises(ValueError, match="No GP model trained"):
        retoucher.reconstruct_image()


def test_image_without_missing_pixels():
    """Test reconstructing an image with no missing pixels."""
    data = np.random.rand(10, 10)
    image = Image(data=data)
    retoucher = Retoucher()
    retoucher.load_image(image)
    retoucher.learn_image(max_iters=1)
    reconstructed_image = retoucher.reconstruct_image()

    assert np.array_equal(reconstructed_image.data, image.data)
