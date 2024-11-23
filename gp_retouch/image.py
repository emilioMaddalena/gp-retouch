from typing import Dict, Optional

import numpy as np

from .image_processor import ImageProcessor


class Image:
    def __init__(
        self,
        data: np.ndarray,
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """_summary_

        Args:
            data (np.ndarray): _description_
            filename (Optional[str], optional): _description_. Defaults to None.
        """
        self.data = data
        self.filename = filename
        self.metadata = metadata or {}

    def get_shape(self) -> tuple[int, int, int]:
        """Returns the shape of the image.

        Returns:
            tuple[int, int, int]: the shape of the image.
        """
        return self.data.shape

    def save(self, filepath: str):
        """_summary_

        Args:
            filepath (str): _description_
        """
        pass

    def to_grayscale(self):
        """Converts the image from RGB to grayscale."""
        if len(self.data.shape) == 3 and self.data.shape[2] == 3:  
            self.data = np.mean(self.data, axis=2).astype(np.uint8)
