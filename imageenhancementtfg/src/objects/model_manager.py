import logging
import gradio as gr

from models.lens_distortion.lens_distortion import LensDistortion
from models.white_balance.white_balance import WhiteBalance
from models.low_light.code.lowlight import LowLight
from models.denoise_deblur.denoise import Denoise
from models.denoise_deblur.deblur import Deblur
from models.sky_replace.sky_replace import SkyReplace


class ModelManager:
    """
    Manager class for loading and handling various deep learning models.

    Attributes
    ----------
    wb_model : WhiteBalance or None
        Instance of the WhiteBalance model.
    ll_model : LowLight or None
        Instance of the LowLight model.
    denoise_model : Denoise or None
        Instance of the Denoise model.
    sky_model : SkyReplace or None
        Instance of the SkyReplace model.
    deblur_model : Deblur or None
        Instance of the Deblur model.
    """
    def __init__(self):
        """
        Constructs a ModelManager instance with attributes initialized to None.
        """
        self.wb_model = None
        self.ll_model = None
        self.denoise_model = None
        self.sky_model = None
        self.deblur_model = None
        self.lens_distortion_model = None

    def load_ll_model(self) -> None:
        """
        Load the Low Light model.load.
        """
        logging.info("Loading Low Light model")
        self.ll_model = LowLight()
        self.ll_model.load_model()
        logging.info("Low Light model loaded")

    def load_denoise_model(self) -> None:
        """
        Load the Denoising model.
        """
        logging.info("Loading Denoising model")
        self.denoise_model = Denoise()
        self.denoise_model.load_model()
        logging.info("Denoising model loaded")

    def load_deblur_model(self) -> None:
        """
        Load the Deblurring model.
        """
        logging.info("Loading Deblurring model")
        self.deblur_model = Deblur()
        self.deblur_model.load_model()
        logging.info("Deblurring model loaded")

    def load_wb_model(self) -> None:
        """
        Load the White Balance model.
        """
        logging.info('Loading White Balance model')
        self.wb_model = WhiteBalance()
        self.wb_model.load_model()
        logging.info('White Balance model loaded')

    def load_lens_distortion_model(self) -> None:
        """
        Load the Lens Distortion model.
        """
        logging.info('Loading Lens Distortion model')
        self.lens_distortion_model = LensDistortion()
        self.lens_distortion_model.load_model()
        logging.info('Lens Distortion model loaded')

    def load_sky_model(self) -> None:
        """
        Load the Sky Replacement model.
        """
        logging.info('Loading Sky Replacement model')
        self.sky_model = SkyReplace()
        self.sky_model.load_model()
        logging.info('Sky Replacement model loaded')

    def load_all_models(self, progress: gr.Progress = gr.Progress()) -> None:
        """
        Load all models managed by this ModelManager.

        Parameters
        ----------
        progress : gr.Progress, optional
            Progress indicator for loading models (default is gr.Progress()).
        """
        logging.info('Loading all models')

        models = [
            ("Low Light model", self.load_ll_model),
            ("Denoising model", self.load_denoise_model),
            ("Deblurring model", self.load_deblur_model),
            ("White Balance model", self.load_wb_model),
            ("Sky Replacement model", self.load_sky_model),
        ]

        step_increment = 1.0 / len(models)  # Progress increment for each model
        current_progress = 0

        for model_name, load_function in models:
            progress(current_progress, desc=f"Loading {model_name}...")
            load_function()

            current_progress += step_increment

        logging.info("All models loaded")
