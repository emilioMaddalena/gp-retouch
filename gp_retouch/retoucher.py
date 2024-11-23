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
        """_summary_

        Args:
            path (Union[str, Path]): _description_
            grayscale (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_
        """
        if grayscale:
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Failed to load image from {path}")
        print("Image loaded successfully.")

    def fill_nans(self, model, **model_args):
        """_summary_

        Args:
            model (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
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
