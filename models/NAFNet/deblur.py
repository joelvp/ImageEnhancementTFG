
import numpy as np
import cv2

from models.NAFNet.basicsr.utils.options import parse
from models.NAFNet.basicsr.models import create_model
from models.NAFNet.basicsr.utils import img2tensor, tensor2img
from src.objects.model import Model


class Deblur(Model):
    def __init__(self, config_path="data/model_config/deblur_config.yml", device='cuda'):
        super().__init__(config_path=config_path, device=device, model_path=None)

    def load_model(self):
        opt = parse(self.config_path, is_train=False)
        opt['dist'] = False
        self.model = create_model(opt)

    def process_image(self, input_image):
        #TODO Revisar esto
        # Comprobar si el tipo de datos no es float32 y convertirlo si es necesario
        if input_image.dtype != np.float32:
            input_image = input_image.astype(np.float32) / 255.

        img = img2tensor(input_image, bgr2rgb=False, float32=True)

        ## Run inference
        self.model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if self.model.opt['val'].get('grids', False):
            self.model.grids()

        self.model.test()

        if self.model.opt['val'].get('grids', False):
            self.model.grids_inverse()

        visuals = self.model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])

        return cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)

