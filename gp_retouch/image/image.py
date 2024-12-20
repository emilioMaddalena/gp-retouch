from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


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
        self.data = data.astype(float)
        self.metadata = metadata or {}

    @property
    def is_grayscale(self) -> bool:  # noqa: D102
        return True if len(self.data.shape) == 2 else False

    @property
    def is_rgb(self) -> bool:  # noqa: D102
        return True if (len(self.data.shape) == 3) and (self.data.shape[2] == 3) else False

    @property
    def shape(self) -> tuple:  # noqa: D102
        return self.data.shape
    
    @property
    def height(self) -> tuple:  # noqa: D102
        if self.is_grayscale:
            height, _ = self.data.shape
        elif self.is_rgb:
            height, _, _ = self.data.shape
        return height
    
    @property
    def width(self) -> tuple:  # noqa: D102
        if self.is_grayscale:
            _, width = self.data.shape
        elif self.is_rgb:
            _, width, _ = self.data.shape
        return width

    @property
    def is_incomplete(self) -> bool:  # noqa: D102
        return np.isnan(self.data).any()

    def get_completeness_ratio(self) -> float:
        """Compute the percentage of pixels that are not nans."""
        return 1 - (np.sum(np.isnan(self.data)) / self.data.size)

    def save(self, filepath: str):  # noqa: D102
        pass

    def plot(self):  # noqa: D102
        if self.is_rgb:
            plt.imshow(self.data.astype(np.uint8))
        else:
            plt.imshow(self.data, cmap='gray', vmin=0, vmax=255)
        plt.axis("off")  
        plt.show()
