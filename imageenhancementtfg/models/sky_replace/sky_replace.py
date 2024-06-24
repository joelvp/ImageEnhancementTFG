import os

from tenacity import retry, stop_after_attempt

from models.sky_replace.networks import * # Cambiar en utils.py el import de la linea 4 por -> from skimage.metrics import structural_similarity as sk_cpt_ssim
from models.sky_replace.skyboxengine import *
import models.sky_replace.utils as utils
import torch

from models.utils import clear_cuda_cache, reset_gradio_flag, load_config
from src.objects.model import Model

config = load_config('data/config.ini')


class SkyReplace(Model):
    """
    Subclass of Model for replacing sky in images using deep learning.

    Attributes
    ----------
    config_path : str
        Path to the configuration file for SkyReplace.
    device : str
        Device on which the model will run. Should be 'cuda' or 'cpu'.
    config : dict
        Parsed configuration from the JSON file specified by `config_path`.
    in_size_w : int
        Width of the input image size expected by the model.
    in_size_h : int
        Height of the input image size expected by the model.
    net_G : torch.nn.Module
        Generator network for sky replacement.

    Methods
    -------
    load_model()
        Load the sky replacement model.
    process_image(input_image: np.ndarray, background_image: np.ndarray=None) -> np.ndarray
        Process an input image and replace the sky with the given background image (optional).
    _process_image_impl(input_image: np.ndarray, background_image: np.ndarray) -> np.ndarray
        Implementation method for processing the input image and performing sky replacement.
    set_output_size(input_image: np.ndarray)
        Set the output size based on the input image dimensions.
    synthesize(img_HD: np.ndarray, img_HD_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        Synthesize the sky-replaced image and return the synthesized image along with other intermediate results.
    cvtcolor_and_resize(img_HD: np.ndarray) -> np.ndarray
        Convert color space and resize the high-definition input image.

    Notes
    -----
    This class is designed specifically for sky replacement tasks using deep learning techniques.
    """
    def __init__(self, config_path=config['models']['sky_replace_config'], device='cuda'):
        """
        Constructs all the necessary attributes for the SkyReplace object.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file for SkyReplace (default is from config.ini).
        device : str, optional
            Device on which the model will run. Should be 'cuda' or 'cpu' (default is 'cuda').
        """
        super().__init__(config_path=config_path, device=device, model_path=None)
        self.config = utils.parse_config(path_to_json=self.config_path)
        self.in_size_w, self.in_size_h = self.config.in_size_w, self.config.in_size_h
        self.net_G = define_G(input_nc=3, output_nc=3, ngf=64, netG=self.config.net_G).to(self.device)

    def load_model(self) -> None:
        """
        Load the sky replacement model.
        """
        checkpoint = torch.load(os.path.join(self.config.ckptdir, 'best_ckpt.pt'),
                                map_location=device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()

    @retry(stop=stop_after_attempt(3), before=clear_cuda_cache)
    def process_image(self, input_image: np.ndarray, background_image: np.ndarray = None) -> np.ndarray:
        """
        Process an input image and replace the sky with the given background image (optional).

        Parameters
        ----------
        input_image : np.ndarray
            The input image to process.
        background_image : np.ndarray, optional
            The background image to use for replacing the sky (default is None).

        Returns
        -------
        np.ndarray
            The processed image with the sky replaced.
        """
        image = self._process_image_impl(input_image, background_image)
        reset_gradio_flag()
        return image

    def _process_image_impl(self, input_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        """
        Implementation method for processing the input image and performing sky replacement.

        Parameters
        ----------
        input_image : np.ndarray
            The input image to process.
        background_image : np.ndarray
            The background image to use for replacing the sky.

        Returns
        -------
        np.ndarray
            The processed image with the sky replaced.
        """
        self.set_output_size(input_image)

        self.config.out_size_w = self.out_size_w
        self.config.out_size_h = self.out_size_h

        self.skyboxengine = SkyBox(self.config, background_image)

        img_HD = self.cvtcolor_and_resize(input_image)
        img_HD_prev = img_HD

        syneth = self.synthesize(img_HD, img_HD_prev)
        syneth = np.array(255.0 * syneth, dtype=np.uint8)

        return syneth

    def set_output_size(self, input_image: np.ndarray) -> None:
        """
        Set the output size based on the input image dimensions.

        Parameters
        ----------
        input_image : np.ndarray
            The input image to process.
        """
        self.out_size_h, self.out_size_w,_ = input_image.shape

    def synthesize(self, img_HD: np.ndarray, img_HD_prev: np.ndarray) -> np.ndarray:
        """
        Synthesize the sky-replaced image.

        Parameters
        ----------
        img_HD : np.ndarray
            The high-definition input image.
        img_HD_prev : np.ndarray
            The previous high-definition input image.

        Returns
        -------
        np.ndarray
            The synthesized sky-replaced image.
        """
        h, w, c = img_HD.shape

        img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))

        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            G_pred = self.net_G(img.to(device))
            G_pred = torch.nn.functional.interpolate(G_pred,
                                                     (h, w),
                                                     mode='bicubic',
                                                     align_corners=False)
            G_pred = G_pred[0, :].permute([1, 2, 0])
            G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)
            G_pred = np.array(G_pred.detach().cpu())
            G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

        skymask = self.skyboxengine.skymask_refinement(G_pred, img_HD)

        syneth = self.skyboxengine.skyblend(img_HD, img_HD_prev, skymask)

        return syneth

    def cvtcolor_and_resize(self, img_HD: np.ndarray) -> np.ndarray:
        """
        Convert color space and resize the high-definition input image.

        Parameters
        ----------
        img_HD : np.ndarray
            The high-definition input image.

        Returns
        -------
        np.ndarray
            The processed high-definition image.
        """
        img_HD = np.array(img_HD / 255., dtype=np.float32)
        return img_HD

