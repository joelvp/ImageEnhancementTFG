import logging
from models.Deep_White_Balance.PyTorch.white_balance import load_wb_model
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
        logging.info('Loading AWB model')
        self.wb_model = load_wb_model()

    def load_sky_model(self):
        logging.info('Loading Sky Replacement model')
        self.sky_model, self.sky_config = load_sky_model()

