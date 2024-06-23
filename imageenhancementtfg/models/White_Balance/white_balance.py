import os
import torch
import numpy as np
from PIL import Image

from models.utils import load_config
from src.objects.model import Model
from models.White_Balance.arch import deep_wb_model, deep_wb_single_task, splitNetworks
from models.White_Balance.utilities.deepWB import deep_wb

config = load_config('data/config.ini')


class WhiteBalance(Model):
    """
    Subclass of Model for improving white balance in images.

    This model enhances the white balance of images using a deep learning-based approach.

    Attributes
    ----------
    model_path : str
        Path to the directory containing the white balance model files.
    device : str
        Device on which the model will run. Should be 'cuda' or 'cpu'.
        If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.

    Methods
    -------
    load_model()
        Load the white balance model.
    _process_image_impl(input_image: np.ndarray, task: str='AWB', maxsize: int=656) -> np.ndarray
        Process an input image to improve its white balance and return the enhanced image.
    _load_awb_model(awb_path: str)
        Load the auto white balance (AWB) model from the specified path.
    _load_split_model(net_path: str)
        Load a split version of the model from the specified path.
    _get_model_instance(model_path: str)
        Get an instance of the appropriate white balance model based on the model path.
    """
    def __init__(self, model_path=config['models']['wb_model'], device='cuda'):
        """
        Constructs all the necessary attributes for the WhiteBalance object.

        Parameters
        ----------
        model_path : str, optional
            Path to the directory containing the white balance model files (default is the path from config.ini).
        device : str, optional
            Device on which the model will run. Should be 'cuda' or 'cpu' (default is 'cuda').
            If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.
        """
        super().__init__(model_path=model_path, device=device, config_path=None)

    def load_model(self) -> None:
        """
        Load the white balance model and initialize its parameters.
        """
        # Load only AWB model
        awb_path = os.path.join(self.model_path, 'net_awb.pth')
        if os.path.exists(awb_path):
            net_awb = self._load_awb_model(awb_path)
        elif os.path.exists(os.path.join(self.model_path, 'net.pth')):
            net_awb = self._load_split_model(os.path.join(self.model_path, 'net.pth'))
        else:
            raise Exception('Model not found!!')
        self.model = net_awb

    def _process_image_impl(self, input_image: np.ndarray, task: str = 'AWB', maxsize: int = 656) -> np.ndarray:
        """
        Process an input image to improve its white balance and return the enhanced image.

        Parameters
        ----------
        input_image : np.ndarray
            The input image to process.
        task : str, optional
            Task to perform ('AWB' for auto white balance) (default is 'AWB').
        maxsize : int, optional
            Maximum size for processing (default is 656).

        Returns
        -------
        np.ndarray
            The enhanced image with improved white balance.
        """
        image = Image.fromarray(np.uint8(input_image))
        out_awb = deep_wb(image, task=task.lower(), net_awb=self.model, device=self.device, s=maxsize)
        return (out_awb * 255).astype(np.uint8)

    def _load_awb_model(self, awb_path: str) -> torch.nn.Module:
        """
        Load the auto white balance (AWB) model from the specified path.

        Parameters
        ----------
        awb_path : str
            Path to the AWB model file.

        Returns
        -------
        torch.nn.Module
            Loaded AWB model.
        """
        net_awb = self._get_model_instance(awb_path)
        net_awb.to(device=self.device)
        net_awb.load_state_dict(torch.load(awb_path, map_location=self.device))
        net_awb.eval()
        return net_awb

    def _load_split_model(self, net_path: str) -> torch.nn.Module:
        """
        Load a split version of the model from the specified path.

        Parameters
        ----------
        net_path : str
            Path to the split model file.

        Returns
        -------
        torch.nn.Module
            Loaded AWB split model.
        """
        net = self._get_model_instance(net_path)
        net.load_state_dict(torch.load(net_path))
        net_awb, _, _ = splitNetworks.splitNetworks(net)
        net_awb.to(device=self.device)
        net_awb.eval()
        return net_awb

    @staticmethod
    def _get_model_instance(self, model_path: str) -> torch.nn.Module:
        """
        Get an instance of the appropriate white balance model based on the model path.

        Parameters
        ----------
        model_path : str
            Path to the white balance model file.

        Returns
        -------
        torch.nn.Module
            Instance of the white balance model.
        """
        if any(filename in model_path for filename in ['net_awb.pth', 'net_t.pth', 'net_s.pth']):
            return deep_wb_single_task.deepWBnet()
        else:
            return deep_wb_model.deepWBNet()


