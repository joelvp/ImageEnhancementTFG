import yaml
import torch
import numpy as np
from PIL import Image
import cv2

from models.NAFNet.basicsr.utils.options import parse
from models.NAFNet.basicsr.models import create_model
from models.NAFNet.basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

def denoising(config="models/NAFNet/options/test/SIDD/NAFNet-width64.yml"):
    # Load the configuration from the .yaml file
    with open(config, 'r') as file:
        opt = yaml.safe_load(file)
    # parse options, set distributed setting, set random seed
    #opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')

    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, output_path)

    print(f'inference {img_path} .. finished. saved to {output_path}')
    
    
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