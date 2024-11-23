from pathlib import Path
from typing import Union

import cv2  
import matplotlib.pyplot as plt
import numpy as np


class Retoucher:
    def __init__(self):
        self.image = None
        self.downscaled_image = None

    def load_image(self, path: Union[str, Path], grayscale: bool = True):
        """
        Load an image from a file path.

        Args:
            path (str): Path to the image file.
            grayscale (bool): Whether to load the image as grayscale.
        """
        if grayscale:
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Failed to load image from {path}")
        print("Image loaded successfully.")

    def downscale_image(self, scale_factor: float):
        """
        Downscale the image by a specified factor.

        Args:
            scale_factor (float): Factor by which to downscale the image.
        """
        if self.image is None:
            raise ValueError("No image loaded. Use 'load_image' first.")
        width = int(self.image.shape[1] * scale_factor)
        height = int(self.image.shape[0] * scale_factor)
        self.downscaled_image = cv2.resize(
            self.image, (width, height), interpolation=cv2.INTER_AREA
        )
        print("Image downscaled successfully.")

    def fill_nans(self, model, **model_args):
        """
        Fill NaN values in the image using an external machine learning model.

        Args:
            model (callable): A callable that accepts an image with NaNs and returns the filled image.
            model_args (dict): Additional arguments for the model.
        """
        if self.downscaled_image is None:
            raise ValueError(
                "No downscaled image available. Use 'downscale_image' first."
            )
        if not np.isnan(self.downscaled_image).any():
            print("No NaNs found in the image. Nothing to fill.")
            return self.downscaled_image

        # Apply the model to fill NaNs
        filled_image = model(self.downscaled_image, **model_args)
        self.downscaled_image = filled_image
        print("NaNs filled using the provided model.")
        return filled_image

    def visualize_images(self):
        """
        Visualize the original and downscaled images.
        """
        if self.image is None:
            raise ValueError("No image loaded. Use 'load_image' first.")
        if self.downscaled_image is None:
            raise ValueError(
                "No downscaled image available. Use 'downscale_image' first."
            )

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(self.image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Downscaled Image")
        plt.imshow(self.downscaled_image, cmap="gray")
        plt.axis("off")

        plt.show()
