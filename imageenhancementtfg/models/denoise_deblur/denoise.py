import numpy as np
import cv2

from models.utils import load_config
from src.objects.model import Model
from models.denoise_deblur.basicsr.utils.options import parse
from models.denoise_deblur.basicsr.models import create_model
from models.denoise_deblur.basicsr.utils import img2tensor, tensor2img

config = load_config('data/config.ini')


class Denoise(Model):
    """
    Subclass of Model for denoising images.

    This model aims to reduce noise in images using a specific deep learning model.

    Attributes
    ----------
    config_path : str
        Path to the configuration file for the denoising model.
    device : str
        Device on which the model will run. Should be 'cuda' or 'cpu'.
        If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.

    Methods
    -------
    load_model()
        Load the denoising model.
    _process_image_impl(input_image: np.ndarray) -> np.ndarray
        Process a noisy input image and return the denoised image.
    """
    def __init__(self, config_path=config['models']['denoise_config'], device='cuda'):
        """
        Constructs all the necessary attributes for the Denoise object.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file for the denoising model (default is the path from config.ini).
        device : str, optional
            Device on which the model will run. Should be 'cuda' or 'cpu' (default is 'cuda').
            If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.
        """
        super().__init__(config_path=config_path, device=device, model_path=None)

    def load_model(self) -> None:
        """
        Load the denoising model and initialize its parameters.
        """
        opt = parse(self.config_path, is_train=False)
        opt['dist'] = False
        self.model = create_model(opt)

    def _process_image_impl(self, input_image: np.ndarray) -> np.ndarray:
        """
        Process a noisy input image and return the denoised image.

        Parameters
        ----------
        input_image : np.ndarray
            The noisy input image to process.

        Returns
        -------
        np.ndarray
            The denoised image.
        """
        img_norm = input_image.astype(np.float32) / 255.

        img = img2tensor(img_norm, bgr2rgb=False, float32=True)

        self.model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if self.model.opt['val'].get('grids', False):
            self.model.grids()

        self.model.test()

        if self.model.opt['val'].get('grids', False):
            self.model.grids_inverse()

        visuals = self.model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])

        return cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
