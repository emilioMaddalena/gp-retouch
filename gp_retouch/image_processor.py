import numpy as np
from skimage.transform import resize


class ImageProcessor:
    """
    Handles low-level image processing tasks.
    """

    def downscale(self, data: np.ndarray, factor: float) -> np.ndarray:
        """_summary_

        Args:
            data (np.ndarray): _description_
            factor (float): _description_

        Returns:
            np.ndarray: _description_
        """
        new_shape = (int(data.shape[0] * factor), int(data.shape[1] * factor))
        return resize(data, new_shape, anti_aliasing=True)

    def convert_to_grayscale(self, data: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            data (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        if len(data.shape) == 3 and data.shape[2] == 3:  # RGB image
            return np.mean(data, axis=2).astype(np.uint8)
        return data  # Already grayscale
