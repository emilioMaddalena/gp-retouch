import copy

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from .image import Image


class ImageProcessor:
    """Handles low-level image processing tasks."""

    @staticmethod
    def downscale(original_image: Image, factor: float) -> Image:
        """Downscale an image by factor."""
        if factor <= 0:
            raise ValueError("The downscale factor must be strictly greater than zero.")
        if factor >= 1:
            raise ValueError("The downscale factor must be strictly smaller than one.")
        image = copy.deepcopy(original_image)
        # process the dimensions relative to the size
        new_shape = (int(image.shape[0] * factor), int(image.shape[1] * factor))
        if image.is_grayscale:
            downscaled_data = resize(image.data, new_shape, anti_aliasing=True)
        if image.is_rgb:
            downscaled_data = resize(image.data, new_shape + (3,), anti_aliasing=True)
        image.data = downscaled_data
        return image

    @staticmethod
    def convert_to_grayscale(image: Image) -> Image:
        """_summary_.

        Args:
            image (Image): _description_

        Returns:
            Image: _description_
        """
        image_copy = copy.deepcopy(image)
        if image_copy.is_grayscale:
            return image
        if image_copy.is_rgb:
            image_copy.data = np.mean(image_copy.data, axis=2).astype(np.uint8)
            return image_copy

    @staticmethod
    def convert_to_rgb(image: Image) -> np.ndarray:  # noqa: D102
        pass

    @staticmethod
    def add_noise(image: Image, method: str = "gaussian", **kwargs) -> Image:
        """Taint the image with the noise of your choice.

        Args:
            image (Image): the input.
            method (str, optional): the noise method of choice. Defaults to "gaussian".
            **kwargs: options to be passed to the individual methods.

        Returns:
            Image: the result.
        """
        def add_gaussian_noise(image: Image, variance: float) -> Image:
            """Taint the image with Gaussian noise."""
            if variance < 0:
                raise ValueError("Variance must be non-negative.")

            # Generate Gaussian noise
            mean = 0
            std_dev = np.sqrt(variance)
            noise = np.random.normal(mean, std_dev, image.data.shape)
            # Add noise to the image
            image.data = image.data + noise
            # Ensure it stays bounded
            image.data = np.clip(image.data, 0, 255)
            return image
    
        methods = {
            "gaussian": add_gaussian_noise,
        }

        if method not in methods:
            raise ValueError(f"Invalid method '{method}'. Valid ones: {list(methods.keys())}")
        
        # Dispatch the requested method
        return methods[method](copy.deepcopy(image), **kwargs)

    @staticmethod
    def drop_pixels(image: Image, ratio: float, method: str = "rnd") -> Image:
        """Drop pixels from the image (turn them into NaNs) using the specified method.

        This method does not transform the image in place.

        Args:
            image (Image): The image to be transformed.
            ratio (float): The ratio of points to be dropped.
            method (str, optional): The method to use for dropping pixels. Defaults to "rnd".

        Returns:
            Image: A new image with some pixels dropped.
        """
        if not (0 < ratio < 1):
            raise ValueError("ratio must be a greater than 0 and smaller than 1.")

        new_image = copy.deepcopy(image)

        # Dispatcher for pixel dropping methods
        methods = {
            "rnd": ImageProcessor._drop_pixels_random,
            "rectangle": ImageProcessor._drop_pixels_rectangle,
            "spiral": ImageProcessor._drop_pixels_spiral,
        }

        if method not in methods:
            raise ValueError(f"Invalid method '{method}'. Valid ones: {list(methods.keys())}")

        return methods[method](new_image, ratio)

    @staticmethod
    def _drop_pixels_random(image: Image, ratio: float) -> Image:
        """Drop pixels randomly from the image."""
        n, m = image.shape[0], image.shape[1]
        num_pixels_drop = round(n * m * ratio)
        indices_drop = np.random.choice(n * m, size=num_pixels_drop, replace=False)
        row_drop, col_drop = np.unravel_index(indices_drop, (n, m))

        if image.is_rgb:
            image.data[row_drop, col_drop, :] = np.nan
        elif image.is_grayscale:
            image.data[row_drop, col_drop] = np.nan

        return image

    @staticmethod
    def _drop_pixels_rectangle(image: Image, ratio: float) -> Image:
        """Drop pixels in a rectangular region."""
        height, width = image.height, image.width
        rect_height = int(height * ratio)
        rect_width = int(width * ratio)
        x = np.random.randint(0, width - rect_width)
        y = np.random.randint(0, height - rect_height)

        if image.is_grayscale:
            image.data[y : y + rect_height, x : x + rect_width] = np.nan
        elif image.is_rgb:
            image.data[y : y + rect_height, x : x + rect_width, :] = np.nan

        return image

    @staticmethod
    def _drop_pixels_spiral(image: Image, ratio: float) -> Image:
        """Drop pixels in a spiral pattern."""
        turns = 3
        n, m = image.height, image.width
        center_y, center_x = n // 2, m // 2 

        max_radius = np.min([n, m]) // 2 * ratio
        y, x = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
        y_shifted = y - center_y
        x_shifted = x - center_x

        r = np.sqrt(x_shifted**2 + y_shifted**2)
        theta = np.arctan2(y_shifted, x_shifted)

        spiral_pattern = (theta + turns * 2 * np.pi * (r / max_radius)) % (2 * np.pi)
        mask = (spiral_pattern < np.pi) & (r < max_radius)

        image.data[mask] = np.nan

        return image
