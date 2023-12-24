import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import logging
from models.SkyAR.networks import * # Cambiar en utils.py el import de la linea 4 por -> from skimage.metrics import structural_similarity as sk_cpt_ssim
from models.SkyAR.skyboxengine import *
import models.SkyAR.utils as utils
import torch

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SkyFilter():

    def __init__(self, config):

        self.ckptdir = config.ckptdir

        self.in_size_w, self.in_size_h = config.in_size_w, config.in_size_h

        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG=config.net_G).to(device)
        self.load_model()

        self.output_img_list = []

    def set_output_size(self, input_image):
        self.out_size_h, self.out_size_w,_ = input_image.shape


    def load_model(self):
        # load pretrained sky matting model
        logging.info('loading the best checkpoint...')
        checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'),
                                map_location=device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()


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
        
    def process_gradio_image(self, input_image, sky_image, config):
        
        self.set_output_size(input_image)
        
        config.out_size_w = self.out_size_w
        config.out_size_h = self.out_size_h
        
        self.skyboxengine = SkyBox(config, sky_image)
        
        img_HD = self.cvtcolor_and_resize(input_image)
        img_HD_prev = img_HD
        
        syneth, _, _ = self.synthesize(img_HD, img_HD_prev)
        syneth = np.array(255.0 * syneth, dtype=np.uint8)
        
        return syneth
        
def load_sky_model():
    config = utils.parse_config(path_to_json='./models/SkyAR/config/gradio_image.json')
    sf = SkyFilter(config)
    
    return sf, config
        
def sky_replace_gui(input_image, sky_image, model, config):
    
    image_sky = model.process_gradio_image(input_image, sky_image, config)
    
    return image_sky
    
