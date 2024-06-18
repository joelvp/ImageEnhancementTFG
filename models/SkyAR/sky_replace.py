import os
import logging
from models.SkyAR.networks import * # Cambiar en utils.py el import de la linea 4 por -> from skimage.metrics import structural_similarity as sk_cpt_ssim
from models.SkyAR.skyboxengine import *
import models.SkyAR.utils as utils
import torch

from src.objects.model import Model

import configparser
config = configparser.ConfigParser()
config.read('data\config.ini')

class SkyReplace(Model):
    def __init__(self, config_path=config['models']['sky_replace_config'], device='cuda'):
        super().__init__(config_path=config_path, device=device, model_path=None)
        self.config = utils.parse_config(path_to_json=self.config_path)
        self.in_size_w, self.in_size_h = self.config.in_size_w, self.config.in_size_h
        self.net_G = define_G(input_nc=3, output_nc=3, ngf=64, netG=self.config.net_G).to(self.device)

    def load_model(self):
        checkpoint = torch.load(os.path.join(self.config.ckptdir, 'best_ckpt.pt'),
                                map_location=device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()

    def process_image(self, input_image, background_image):
        self.set_output_size(input_image)

        self.config.out_size_w = self.out_size_w
        self.config.out_size_h = self.out_size_h

        self.skyboxengine = SkyBox(self.config, background_image)

        img_HD = self.cvtcolor_and_resize(input_image)
        img_HD_prev = img_HD

        syneth, _, _ = self.synthesize(img_HD, img_HD_prev)
        syneth = np.array(255.0 * syneth, dtype=np.uint8)

        return syneth

    def set_output_size(self, input_image):
        self.out_size_h, self.out_size_w,_ = input_image.shape

    def synthesize(self, img_HD, img_HD_prev):

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

        return syneth, G_pred, skymask

    def cvtcolor_and_resize(self, img_HD):

        img_HD = np.array(img_HD / 255., dtype=np.float32)

        return img_HD

