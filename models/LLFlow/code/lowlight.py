
        
import torch
from models.LLFlow.code.test_unpaired import auto_padding, hiseq_color_cv2_img, load_LLFlow_model, rgb, t
from src.objects.model import Model

import configparser
config = configparser.ConfigParser()
config.read('data\config.ini')


class LowLight(Model):
    def __init__(self, config_path=config['models']['lowlight_config'], device='cuda'):
        super().__init__(config_path=config_path, device=device, model_path=None)
        self.opt = {}

    def load_model(self):
        self.model, self.opt = load_LLFlow_model(self.config_path)
        self.model.netG = self.model.netG.cuda()

    def process_image(self, input_image):

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

