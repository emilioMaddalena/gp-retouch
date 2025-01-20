import copy

import numpy as np
from skimage.transform import resize

from .image import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE, PIXEL_DATA_TYPE, Image


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
            image (Image): The input image.
            method (str, optional): The noise method of choice: gausian, salt_and_pepper,
                                    speckle, or uniform. Defaults to "gaussian".
            **kwargs: Options to be passed to the individual methods.

        Returns:
            Image: The result with added noise.
        """
        methods = {
            "gaussian": ImageProcessor._add_gaussian_noise,
            "salt_and_pepper": ImageProcessor._add_salt_and_pepper_noise,
            "speckle": ImageProcessor._add_speckle_noise,
            "uniform": ImageProcessor._add_uniform_noise,
        }

        if method not in methods:
            raise ValueError(f"Invalid method '{method}'. Valid methods: {list(methods.keys())}")

        # Dispatch the requested method
        return methods[method](copy.deepcopy(image), **kwargs)
    
    @staticmethod
    def _add_gaussian_noise(image: Image, variance: float) -> Image:
        """Taint the image with Gaussian noise."""
        if variance < 0:
            raise ValueError("Variance must be non-negative.")
        
        # Generate Gaussian noise
        mean = 0
        std_dev = np.sqrt(variance)
        noise = np.random.normal(mean, std_dev, image.data.shape)
        # Add noise to the image
        image.data = ImageProcessor._conform_to_image_data_reqs(image.data + noise)
        return image
    
    @staticmethod
    def _add_salt_and_pepper_noise(image: Image, amount: float, salt_ratio: float = 0.5) -> Image:
        """Taint the image with salt-and-pepper noise."""
        if not (0 < amount < 1):
            raise ValueError("Amount must be between 0 and 1.")
        if not (0 <= salt_ratio <= 1):
            raise ValueError("Salt ratio must be between 0 and 1.")

        num_pixels = np.prod(image.data.shape[:2])
        num_salt = int(amount * num_pixels * salt_ratio)
        num_pepper = int(amount * num_pixels * (1 - salt_ratio))

        # Add salt noise
        salt_coords = tuple(
            np.random.randint(0, i, num_salt) for i in image.data.shape[:2]
        )
        image.data[salt_coords] = 255
        # Add pepper noise
        pepper_coords = tuple(
            np.random.randint(0, i, num_pepper) for i in image.data.shape[:2]
        )
        image.data[pepper_coords] = 0
        return image

    @staticmethod
    def _add_speckle_noise(image: Image, variance: float) -> Image:
        """Taint the image with speckle noise."""
        if variance < 0:
            raise ValueError("Variance must be non-negative.")

        noise = np.random.normal(0, np.sqrt(variance), image.data.shape)
        image.data = ImageProcessor._conform_to_image_data_reqs(image.data + image.data * noise)
        return image

    @staticmethod
    def _add_uniform_noise(image: Image, intensity: float) -> Image:
        """Taint the image with uniform noise."""
        if intensity < 0:
            raise ValueError("Intensity must be non-negative.")

        noise = np.random.uniform(-intensity, intensity, image.data.shape)
        image.data = ImageProcessor._conform_to_image_data_reqs(image.data + noise)
        return image

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

    @staticmethod
    def _conform_to_image_data_reqs(data: np.ndarray) -> np.ndarray:
        """"Ensure compliance with the Image @data.setter method.

        Args:
            data (np.ndarray): Input data to be transformed.

        Returns:
            np.ndarray: Data that can be used by Image.
        """
        data = np.clip(data, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
        data = data.astype(PIXEL_DATA_TYPE)
        return data