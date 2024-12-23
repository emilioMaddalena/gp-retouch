import GPy
import numpy as np

from ..image.image import Image

MIN_NOISE_VAR = 1e-6
VISUAL_EPSILON = 1


class Artist:
    """A class to draw images of kernels."""

    @staticmethod
    def draw_kernels(
        kernel: GPy.kern.Kern, num_centers: int, image_size: tuple = (100, 100)
    ) -> Image:
        """Create an image based on a kernel and a number of random centers.

        Args:
            kernel (GPy.kern.Kern): the kernel to be used.
            num_centers (int): the desired number of random centers.
            image_size (tuple, optional): Defaults to (100, 100).

        Returns:
            Image: the returned image object.
        """
        # Create GP model based on random centers (lying on the image grid)
        grid_points, _ = Artist._get_grid_coords(image_size)
        x_idxs = np.random.choice(grid_points.shape[0], num_centers, replace=False)
        x = grid_points[x_idxs]
        y = np.random.rand(num_centers, 1) * 255  
        gp = GPy.models.GPRegression(x, y, kernel, noise_var=MIN_NOISE_VAR)

        # Predict values on the grid
        grid_coords, _ = Artist._get_grid_coords(image_size)
        mean, _ = gp.predict(grid_coords)
        mean_normalized = 255 * (mean - mean.min()) / (mean.max() - mean.min())
        return Image(data=mean_normalized.reshape(image_size))


    @staticmethod
    def _get_grid_coords(image_size: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Produce all grid points given an image size.

        Args:
            image_size (tuple): the size of the image, e.g. (100, 80).

        Returns:
            tuple: grid_coordinates in their raw form and normalized grid coordinates.
        """
        grid_x, grid_y = np.meshgrid(
            np.arange(image_size[0]), np.arange(image_size[1]), indexing="ij"
        )
        grid_coords = np.c_[grid_x.ravel(), grid_y.ravel()]
        grid_coords_normalized = grid_coords / np.array([image_size[0], image_size[1]])
        return grid_coords, grid_coords_normalized
