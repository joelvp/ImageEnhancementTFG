import logging
import gradio as gr
from models.White_Balance.white_balance import WhiteBalance
from models.LLFlow.code.lowlight import load_ll_model
from models.NAFNet.deblur import load_deblurring_model
from models.NAFNet.denoise import load_denoising_model
from models.SkyAR.sky_replace import load_sky_model


class ModelManager:
    def __init__(self):
        self.wb_model = None
        self.ll_model = None
        self.denoise_model = None
        self.sky_model = None
        self.deblur_model = None
        
    def load_ll_model(self):
        logging.info("Loading Low Light model")
        self.ll_model, self.opt_ll = load_ll_model()
        
    def load_denoise_model(self):
        logging.info("Loading Denoising model")
        self.denoise_model = load_denoising_model()
        
    def load_deblur_model(self):
        logging.info("Loading Deblurring model")
        self.deblur_model = load_deblurring_model()

    def load_wb_model(self):
        logging.info('Loading White Balance model')
        self.wb_model = WhiteBalance()
        self.wb_model.load_model()

    def load_sky_model(self):
        logging.info('Loading Sky Replacement model')
        self.sky_model, self.sky_config = load_sky_model()

    def load_all_models(self, progress=gr.Progress()):
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
            # Load the model
            load_function()

            current_progress += step_increment
            logging.info(f"{model_name} loaded")

        logging.info("All models loaded")


