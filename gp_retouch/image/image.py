from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .image_processor import ImageProcessor


class Image:
    """Represents an image and provides the user basic manipulation methods.

    It encapsulates an image data and metadata, and provides manipulation tools such as 
    rescaling it, converting it to greyscale, plotting and saving it, etc.
    """

    def __init__(
        self,
        path_to_image: Union[Path, str],
        metadata: Optional[Dict] = None,
    ):
        """Load the image.

        Args:
            path_to_image (np.ndarray): the image's actual data.
            metadata (Optional[Dict], optional): any extra info about the data.
        """
        self.data = data
        self.filename = filename
        self.metadata = metadata or {}

    def get_shape(self) -> tuple[int, int, int]:
        """Return the shape of the image.

        Returns:
            tuple[int, int, int]: the shape of the image.
        """
        return self.data.shape

    def save(self, filepath: str):
        """_summary_.

        Args:
            filepath (str): _description_
        """
        pass

    def to_grayscale(self):
        """Convert the image from RGB to grayscale."""
        if len(self.data.shape) == 3 and self.data.shape[2] == 3:
            self.data = np.mean(self.data, axis=2).astype(np.uint8)
