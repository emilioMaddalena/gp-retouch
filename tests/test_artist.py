import GPy
import numpy as np
import pytest

from gp_retouch.artist.artist import Artist
from gp_retouch.image.image import Image


@pytest.mark.parametrize(
    "kernel, num_centers, image_size",
    [
        (GPy.kern.RBF(input_dim=2, variance=10.0, lengthscale=3.0), 2, (100, 100)),
        (GPy.kern.Linear(input_dim=2, variances=10.0), 5, (50, 155)),
        (GPy.kern.Matern32(input_dim=2, variance=1.0, lengthscale=10.0), 3, (11, 200)),
    ]
)
def test_draw_kernels(kernel, num_centers, image_size):  # noqa: D103
    image = Artist.draw_kernels(kernel, num_centers, image_size)

    assert isinstance(image, Image)
    assert image.data.shape == image_size


@pytest.mark.parametrize(
    "image_size",
    [
        ((100, 100)),
        ((50, 101)),
        ((99, 200)),
    ],
)
def test_get_grid_coords(image_size):  # noqa: D103
    grid_coords, grid_coords_normalized = Artist._get_grid_coords(image_size)

    assert grid_coords.shape == (image_size[0] * image_size[1], 2)
    assert grid_coords_normalized.shape == (image_size[0] * image_size[1], 2)
    assert np.all(grid_coords_normalized >= 0) and np.all(grid_coords_normalized <= 1)
