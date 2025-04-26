from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255
PIXEL_DATA_TYPE = float


class Image:
    """Represents an image and provides the user basic manipulation methods.

    It encapsulates an image data and metadata, and provides manipulation tools such as
    rescaling it, converting it to greyscale, plotting and saving it, etc.
    """

    def __init__(
        self,
        data: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        """Load the image.

        Args:
            data (np.ndarray): the image's actual data.
            metadata (Optional[Dict], optional): any extra info about the data.
        """
        self._data = None
        self.data = data
        self.metadata = metadata or {}

    @property
    def data(self) -> np.ndarray:
        """Get the image data."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Set the image data, ensuring values are between 0 and 255."""
        if not np.all(np.isnan(value) | ((MIN_PIXEL_VALUE <= value) & (value <= MAX_PIXEL_VALUE))):
            raise ValueError("All values in the image data must be between 0 and 255, or NaNs.")
        self._data = value.astype(PIXEL_DATA_TYPE)

    @property
    def is_grayscale(self) -> bool:
        """Check if the image is grayscale."""
        return True if len(self.data.shape) == 2 else False

    @property
    def is_rgb(self) -> bool:
        """Check if the image is RGB."""
        return True if (len(self.data.shape) == 3) and (self.data.shape[2] == 3) else False

    @property
    def shape(self) -> tuple:
        """Get the image shape."""
        return self.data.shape

    @property
    def height(self) -> tuple:
        """Get the image height."""
        if self.is_grayscale:
            height, _ = self.data.shape
        elif self.is_rgb:
            height, _, _ = self.data.shape
        return height

    @property
    def width(self) -> tuple:
        """Get the image width."""
        if self.is_grayscale:
            _, width = self.data.shape
        elif self.is_rgb:
            _, width, _ = self.data.shape
        return width

    @property
    def is_incomplete(self) -> bool:
        """Check if the image has any NaN values."""
        return np.isnan(self.data).any()

    def get_completeness_ratio(self) -> float:
        """Compute the percentage of pixels that are not nans."""
        return 1 - (np.sum(np.isnan(self.data)) / self.data.size)

    def plot(self, plot_3d: bool = False):
        """Plot the image either in 2D or 3D (only for grayscale images).

        Args:
            plot_3d (bool, optional): Enables 3D interactive plotting. Defaults to False.
        """
        if plot_3d:
            if self.is_rgb:
                raise ValueError("Cannot 3D plot an RGB image.")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            x, y = np.meshgrid(np.arange(self.data.shape[1]), np.arange(self.data.shape[0]))
            if self.is_rgb:
                ax.plot_surface(
                    x,
                    y,
                    self.data[:, :, 0],
                    rstride=1,
                    cstride=1,
                    facecolors=self.data / MAX_PIXEL_VALUE,
                    shade=False,
                )
            else:
                ax.plot_surface(
                    x,
                    y,
                    self.data,
                    cmap="gray",
                    rstride=1,
                    cstride=1,
                    vmin=MIN_PIXEL_VALUE,
                    vmax=MAX_PIXEL_VALUE,
                )
            plt.show()
        else:
            if self.is_rgb:
                plt.imshow(self.data.astype(np.uint8))
            else:
                plt.imshow(self.data, cmap="gray", vmin=MIN_PIXEL_VALUE, vmax=MAX_PIXEL_VALUE)
            plt.axis("off")
            plt.show()
