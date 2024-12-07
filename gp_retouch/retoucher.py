import copy
from typing import Union

import GPy
import numpy as np

from .image.image import Image


class Retoucher:
    """_summary_."""

    def __init__(self):
        """_summary_."""
        pass

    def load_image(self, image: Image):
        """_summary_.

        Args:
            image (Image): _description_
        """
        self.image = copy.deepcopy(image)

    def learn_image(self):
        """Learn the image pixel distribution."""
        pass

    def reconstruct_image(self) -> Image:
        """_summary_.

        Args:
            model (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        image = self.image

        if not image.is_incomplete:
            print("No missing pixels!")
            return image

        # Initialize the reconstructed image
        reconstructed_data = np.zeros_like(image.data)

        # Process each channel separately
        channels = [image.data] if not image.is_rgb else [image.data[:, :, i] for i in range(3)]

        for c_idx, channel in enumerate(channels):
            # Get coordinates and values of non-NaN pixels
            coords = np.argwhere(~np.isnan(channel))
            values = channel[~np.isnan(channel)]

            # Normalize coordinates for better numerical stability
            coords_normalized = coords / np.array([channel.shape[0], channel.shape[1]])

            # Define the kernel and GP model
            kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=0.02)
            gp = GPy.models.GPRegression(coords_normalized, values[:, None], kernel, noise_var=1e-6)

            # Generate predictions for the entire image grid
            grid_x, grid_y = np.meshgrid(
                np.arange(channel.shape[0]), np.arange(channel.shape[1]), indexing="ij"
            )
            grid_coords = np.c_[grid_x.ravel(), grid_y.ravel()]
            grid_coords_normalized = grid_coords / np.array([channel.shape[0], channel.shape[1]])

            mean, _ = gp.predict(grid_coords_normalized)
            reconstructed_channel = mean.reshape(channel.shape)

            # Assign the reconstructed channel to the output
            if image.is_rgb:
                reconstructed_data[:, :, c_idx] = reconstructed_channel
            else:
                reconstructed_data = reconstructed_channel

        return Image(reconstructed_data)