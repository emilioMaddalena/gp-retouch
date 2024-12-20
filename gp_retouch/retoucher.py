import copy

import GPy
import numpy as np

from .image.image import Image


class Retoucher:
    """A class to reconstruct images with missing pixels using Gaussian Processes."""

    def __init__(self, kernel=None, noise_var=1e-6):
        """Initialize the Retoucher with a kernel and noise variance.

        Args:
            kernel (GPy.kern.Kern): A GPy kernel (default: RBF kernel).
            noise_var (float): Noise variance for the GP model (default: 1e-6).
        """
        self.kernel = kernel or GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=1.0)
        self.noise_var = noise_var
        self.gp = None
        self.image = None

    def load_image(self, image: Image):
        """Load an image into the Retoucher for processing.

        Args:
            image (Image): The image object containing pixel data.
        """
        self.image = copy.deepcopy(image)

    def learn_image(self, **optimization_params):
        """Learn the kernel hyperparameters based on the image's data.

        Args:
            **optimization_params: Key-value parameters passed to the gp.optimize function.
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() to load an image.")

        channels = 3 if self.image.is_rgb else 1
        data = [
            self.image.data[..., channel] if self.image.is_rgb else self.image.data
            for channel in range(channels)
        ]

        self.gp = []
        for channel_data in data:
            coords, values = self._get_non_nan_data(channel_data)
            coords_normalized = coords / np.array([channel_data.shape[0], channel_data.shape[1]])

            # Fit the GP model for the current channel
            gp_channel = GPy.models.GPRegression(
                coords_normalized, values[:, None], self.kernel, noise_var=self.noise_var
            )
            gp_channel.optimize(**optimization_params)
            self.gp.append(gp_channel)

        # If grayscale, store a single GP model instead of a list
        if not self.image.is_rgb:
            self.gp = self.gp[0]

    def reconstruct_image(self) -> Image:
        """Reconstruct the image using the learned GP model.

        Returns:
            Image: The reconstructed image.
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() to load an image.")

        if not self.image.is_incomplete:
            print("No missing pixels!")
            return self.image

        if self.gp is None:
            raise ValueError("No GP model trained. Use learn_image() to train the model.")

        # Handle grayscale or RGB images consistently
        channels = 3 if self.image.is_rgb else 1
        data_list = [
            self.image.data[..., channel] if self.image.is_rgb else self.image.data
            for channel in range(channels)
        ]
        reconstructed_data = np.zeros_like(self.image.data)

        for i, channel_data in enumerate(data_list):
            _, grid_coords_normalized = self._get_grid_coords(channel_data)

            # Predict channel values
            mean, _ = (self.gp[i] if self.image.is_rgb else self.gp).predict(grid_coords_normalized)
            reconstructed_channel = mean.reshape(channel_data.shape)

            # Replace NaNs with predictions
            channel_data[np.isnan(channel_data)] = reconstructed_channel[np.isnan(channel_data)]

            if self.image.is_rgb:
                reconstructed_data[..., i] = channel_data
            else:
                reconstructed_data = channel_data

        return Image(reconstructed_data)
    
    def denoise_image(self, image: Image, factor: float) -> Image:
        """Get rid of unwanted random noise in an image using Gaussian Processes.

        Args:
            image (Image): The input image (grayscale or RGB).
            factor (float): A 0 to 1 factor that controls the filter intensity.

        Returns:
            Image: The denoised image.
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() to load an image.")

        if self.image.is_incomplete:
            print("Image must be compelte!")
            return self.image

        if self.gp is None:
            raise ValueError("No GP model trained. Use learn_image() to train the model.")

        if not 0 <= factor <= 1:
            raise ValueError("Factor must be between 0 and 1.")

        # Prepare the data for denoising
        channels = 3 if image.is_rgb else 1
        data_list = [
            image.data[..., channel] if image.is_rgb else image.data
            for channel in range(channels)
        ]
        denoised_data = np.zeros_like(image.data)

        for i, channel_data in enumerate(data_list):
            _, grid_coords_normalized = self._get_grid_coords(channel_data)

            # Predict channel values
            mean, _ = (self.gp[i] if self.image.is_rgb else self.gp).predict(grid_coords_normalized)
            smoothed_channel = mean.reshape(channel_data.shape)

            # Blend the original and smoothed images based on the factor
            denoised_channel = (1 - factor) * channel_data + factor * smoothed_channel

            if image.is_rgb:
                denoised_data[..., i] = denoised_channel
            else:
                denoised_data = denoised_channel

        return Image(denoised_data)

    @staticmethod
    def sharpen_image(image: Image, factor: float) -> Image:
        """Make a full image sharper (sharper transitions).

        Args:
            image (Image): the input.
            factor (float): a 0 to 1 factor that controls the filter intensity.

        Returns:
            Image: the result.
        """
        pass

    @staticmethod
    def _get_non_nan_data(data):
        """Get the non-NaN coordinates and values from the image data.

        Args:
            data (np.array): The image data.

        Returns:
            Tuple[np.array, np.array]: Non-NaN coordinates and their corresponding values.
        """
        coords = np.argwhere(~np.isnan(data))
        values = data[~np.isnan(data)]
        return coords, values

    @staticmethod
    def _get_grid_coords(data):
        """Get the grid coordinates and their normalized version.

        Args:
            data (np.array): The image data.

        Returns:
            Tuple[np.array, np.array]: Grid coordinates and normalized grid coordinates.
        """
        grid_x, grid_y = np.meshgrid(
            np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
        )
        grid_coords = np.c_[grid_x.ravel(), grid_y.ravel()]
        grid_coords_normalized = grid_coords / np.array([data.shape[0], data.shape[1]])
        return grid_coords, grid_coords_normalized
