import copy

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from .image import Image


class ImageProcessor:
    """Handles low-level image processing tasks."""

    def downscale(self, original_image: Image, factor: float) -> Image:
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

    def convert_to_grayscale(self, image: Image) -> Image:
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

    def convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """_summary_.

        Args:
            image (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        pass

    def print(self, image: np.ndarray):
        """_summary_.

        Args:
            image (np.ndarray): _description_
        """
        plt.imshow(image, cmap="gray")  # Use 'gray' for grayscale images
        plt.axis("off")  # Turn off axis labels and ticks
        plt.show()
