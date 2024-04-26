
import numpy as np
import cv2

from models.NAFNet.basicsr.utils.options import parse
from models.NAFNet.basicsr.models import create_model
from models.NAFNet.basicsr.utils import img2tensor, tensor2img
    
def load_denoising_model(config="models/NAFNet/options/test/SIDD/NAFNet-width64.yml"):
    
    opt = parse(config, is_train=False)
    
    opt['dist'] = False
    model = create_model(opt)
    
    return model

def denoising_gui(input_image, model):

    # Comprobar si el tipo de datos no es float32 y convertirlo si es necesario
    if input_image.dtype != np.float32:
        input_image = input_image.astype(np.float32) / 255.    

    img = img2tensor(input_image, bgr2rgb=False, float32=True)
    
    ## Run inference
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    
    return cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)