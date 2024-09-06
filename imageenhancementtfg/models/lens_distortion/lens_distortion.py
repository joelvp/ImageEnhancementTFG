import os
import numpy as np
import torch

from models.lens_distortion.utils import super_resolution, transform_image
from models.utils import load_config
from src.objects.model import Model
from models.lens_distortion.train.modelNetS_batch1 import EncoderNet, ModelNet
from models.lens_distortion.resample.resampling import rectification

config = load_config('data/config.ini')


class LensDistortion(Model):
    """
        Subclass of Model for lens distortion correction.

        This model corrects lens distortion in images using deep learning-based methods.

        Attributes
        ----------
        model_1_path : str
            Path to the first model used for lens distortion correction.
        model_2_path : str
            Path to the second model used for lens distortion correction.
        model1 : EncoderNet
            Instance of the first neural network model for lens distortion correction.
        model2 : ModelNet
            Instance of the second neural network model for lens distortion correction.

        Methods
        -------
        load_model()
            Load the lens distortion correction models.
        _process_image_impl(input_image: np.ndarray) -> np.ndarray
            Process an input image with lens distortion and return the corrected image.
        """
    def __init__(self, model_path: str = config['models']['lens_distortion_model'], device: str = 'cuda'):
        """
        Constructs all the necessary attributes for the LensDistortion object.

        Parameters
        ----------
        model_path : str, optional
            Path to the directory containing the lens distortion model files (default is the path from config.ini).
        device : str, optional
            Device on which the models will run. Should be 'cuda' or 'cpu' (default is 'cuda').
            If 'cuda' is chosen and CUDA is not available, falls back to 'cpu'.
        """
        super().__init__(model_path=model_path, device=device)
        self.model_1_path = os.path.join(self.model_path, 'model1.pth')
        self.model_2_path = os.path.join(self.model_path, 'model2.pth')
        self.model1 = EncoderNet([1, 1, 1, 1, 2])
        self.model2 = ModelNet('barrel')

    def load_model(self) -> None:
        """
        Load the lens distortion model.
        """
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        self.model1.load_state_dict(torch.load(self.model_1_path))
        self.model2.load_state_dict(torch.load(self.model_2_path))
        self.model1.eval()
        self.model2.eval()

    def _process_image_impl(self, input_image: np.ndarray) -> np.ndarray:
        """
        Process an input image with lens distortion and super resolution and return the corrected image.

        Parameters
        ----------
        input_image : np.ndarray
            The input image with lens distortion to process.

        Returns
        -------
        np.ndarray
            The corrected image.
        """
        im_tensor, im_npy = transform_image(input_image, self.device)
        flow_output = self.model2(self.model1(im_tensor))
        corrected_image, _ = rectification(im_npy, flow_output.data.cpu().numpy()[0])
        enhanced_image = np.array(corrected_image)

        super_image = super_resolution(enhanced_image)

        return super_image
