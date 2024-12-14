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
    def drop_pixels(image: Image, ratio: bool, method: str = "rnd") -> Image:
        """Drop pixels from the image (turn them into nans).

        This method does not transform the image in place.

        Args:
            image (Image): the image to be transformed.
            ratio (bool): the ratio of points to be dropped.
            method (str, optional): TBW.

        Returns:
            Image: a new image with some pixels dropped.
        """
        new_image = copy.deepcopy(image)

        if not (0 < ratio < 1):
            raise ValueError("ratio must be a greater than 0 and smaller than 1.")
        
        if method == "rnd":
            n = new_image.shape[0]
            m = new_image.shape[1]
            num_pixels_drop = round(n * m * ratio)
            indices_drop = np.random.choice(n * m, size=num_pixels_drop, replace=False)
            row_drop, col_drop = np.unravel_index(indices_drop, (n, m))
            if new_image.is_rgb:
                new_image.data[row_drop, col_drop, :] = np.nan
            elif new_image.is_grayscale:
                print(row_drop)
                print(col_drop)
                new_image.data[row_drop, col_drop] = np.nan
            return new_image
        
        elif method == "rectangle":
            height = new_image.height
            width = new_image.width
            # Build the rectangle
            rect_height = int(height * ratio)
            rect_width = int(width * ratio)
            x = np.random.randint(0, width - rect_width)
            y = np.random.randint(0, height - rect_height)
            #x = (width - rect_width) // 2
            #y = (height - rect_height) // 2
            # Fill with NaNs
            if image.is_grayscale:
                new_image.data[y:y+rect_height, x:x+rect_width] = np.nan
            elif image.is_rgb:
                new_image.data[y:y+rect_height, x:x+rect_width, :] = np.nan
            return new_image