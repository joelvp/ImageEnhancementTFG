from typing import Optional

import numpy as np
import torch
import logging
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt

from models.utils import clear_cuda_cache, reset_gradio_flag


class Model(ABC):
    """
    Abstract base class for deep learning models.

    This class provides a structure for loading models and processing images,
    ensuring that derived classes implement the necessary methods.

    Attributes
    ----------
    model_path : str
        Path to the model file.
    config_path : str
        Path to the configuration file.
    device : str
        Device on which the model will run (CPU or CUDA).
    model : Any
        Placeholder for the loaded model.

    Methods
    -------
    load_model()
        Abstract method to load the model. Must be implemented by subclasses.
    process_image(input_image)
        Process an input image using the model with retry logic.
    _process_image_impl(input_image)
        Abstract method for processing an image. Must be implemented by subclasses.
    """
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None, device: str = 'cuda'):
        """
        Constructs all the necessary attributes for the model object.

        Parameters
        ----------
        model_path : Optional[str], optional
            Path to the model file. Default is None.
        config_path : Optional[str], optional
            Path to the configuration file. Default is None.
        device : str, optional
            Device on which the model will run. Should be 'cuda' or 'cpu' (default is 'cuda').
            If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.
        """
        self.device = torch.device('cuda' if device.lower() == 'cuda' and torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        self.model_path = model_path
        self.config_path = config_path
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model. This method must be implemented by subclasses.
        """
        pass

    @retry(stop=stop_after_attempt(3), before=clear_cuda_cache, reraise=True)
    def process_image(self, input_image: np.ndarray) -> np.ndarray:
        """
        Process an input image using the model with retry logic.

        Parameters
        ----------
        input_image : np.ndarray
            The input image to process.

        Returns
        -------
        np.ndarray
            The processed image.
        """
        image = self._process_image_impl(input_image)
        reset_gradio_flag()
        return image

    @abstractmethod
    def _process_image_impl(self, input_image: np.ndarray) -> np.ndarray:
        """
        Process an image. This method must be implemented by subclasses.

        Parameters
        ----------
        input_image : np.ndarray
            The input image to process.

        Returns
        -------
        np.ndarray
            The processed image.
        """
        pass
