import numpy as np
import torch
from models.LLFlow.code.test_unpaired import auto_padding, hiseq_color_cv2_img, load_LLFlow_model, rgb, t
from models.utils import load_config
from src.objects.model import Model

config = load_config('data/config.ini')


class LowLight(Model):
    """
        Subclass of Model for processing low-light images.

        This model aims to enhance very dark images by processing them through a specific deep learning model.

        Attributes
        ----------
        config_path : str
            Path to the configuration file for the low-light model.
        device : str
            Device on which the model will run. Should be 'cuda' or 'cpu'.
            If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.
        Methods
        -------
        load_model()
            Load the low-light model.
        _process_image_impl(input_image: np.ndarray) -> np.ndarray
            Process a low-light input image.

        """
    def __init__(self, config_path=config['models']['lowlight_config'], device='cuda'):
        """
        Constructs all the necessary attributes for the LowLight object.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file for the low-light model (default is the path from config.ini).
        device : str, optional
            Device on which the model will run. Should be 'cuda' or 'cpu' (default is 'cuda').
            If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.
        """
        super().__init__(config_path=config_path, device=device, model_path=None)
        self.opt = {}

    def load_model(self) -> None:
        """
        Load the low-light model and initialize its parameters.
        """
        self.model, self.opt = load_LLFlow_model(self.config_path)
        self.model.netG = self.model.netG.cuda()

    def _process_image_impl(self, input_image: np.ndarray) -> np.ndarray:
        """
        Process a low-light input image.

        Parameters
        ----------
        input_image : np.ndarray
            The low-light input image to process.

        Returns
        -------
        np.ndarray
            The processed enhanced image.
        """
        raw_shape = input_image.shape
        input_image, padding_params = auto_padding(input_image)
        his = hiseq_color_cv2_img(input_image)
        lr_t = t(input_image)
        if self.opt["datasets"]["train"].get("log_low", False):
            lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
        if self.opt.get("concat_histeq", False):
            his = t(his)
            lr_t = torch.cat([lr_t, his], dim=1)
        with torch.cuda.amp.autocast():
            sr_t = self.model.get_sr(lq=lr_t.cuda(), heat=None)

        sr = rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
                 padding_params[2]:sr_t.shape[3] - padding_params[3]])
        assert raw_shape == sr.shape

        return sr

