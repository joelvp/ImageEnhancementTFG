import os
import logging
from PIL import Image
import torch
import numpy as np

from .arch import deep_wb_model, deep_wb_single_task, splitNetworks
from .utilities import utils
from .utilities.deepWB import deep_wb


def white_balance(model_dir='./models/Deep_White_Balance/PyTorch/models',input_dir='../data/demo_images/input/',out_dir='../data/demo_images/output/',
                  task='AWB',target_color_temp=None,mxsize=656,show=False,tosave=True,device='cuda'):
    S = mxsize

    device = torch.device('cuda' if device.lower() == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if target_color_temp:
        assert 2850 <= target_color_temp <= 7500, (
                'Color temperature should be in the range [2850 - 7500], but the given one is %d' % target_color_temp)

        if task.lower() != 'editing':
            raise Exception('The task should be editing when a target color temperature is specified.')

    if tosave and not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if task.lower() == 'all':
        net_awb, net_t, net_s = load_all_models(model_dir, device)
        logging.info("Models loaded !")
        
    elif task.lower() == 'editing':
        t_path = os.path.join(model_dir, 'net_t.pth')
        s_path = os.path.join(model_dir, 'net_s.pth')

        if os.path.exists(t_path) and os.path.exists(s_path):
            net_t = load_model(t_path)
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            
            net_s = load_model(s_path)
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
            logging.info("Models loaded !")
            
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = load_model(os.path.join(model_dir, 'net.pth'))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            _, net_t, net_s = splitNetworks.splitNetworks(net)
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
            
        else:
            raise Exception('Model not found!')
        
    elif task.lower() == 'awb':
        awb_path = os.path.join(model_dir, 'net_awb.pth')
        
        if os.path.exists(awb_path):
            net_awb = load_model(awb_path)
            logging.info(f'Using device {device}')
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
            
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = load_model(os.path.join(model_dir, 'net.pth'))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, _, _ = splitNetworks.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
        else:
            raise Exception('Model not found!')
    else:
        raise Exception("Wrong task! Task should be: 'AWB', 'editing', or 'all'")
    
    imgfiles = process_images(input_dir)
    
    if task.lower() == 'all':  # awb and editing tasks
        all_filter(imgfiles,task,device, net_awb, net_s, net_t, S, tosave, show, out_dir)
        
    elif task.lower() == 'awb':  # awb task
        awb_filter(imgfiles,task,device, net_awb, S, tosave, show, out_dir)
        
    else:  #edit task
        edit_filter(imgfiles,task,device, net_s, net_t, S, tosave, show, target_color_temp, out_dir)
        
######### GUI Function #########

def load_wb_model(model_dir='./models/Deep_White_Balance/PyTorch/models', device='cuda'):

    device = torch.device('cuda' if device.lower() == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')   

    awb_path = os.path.join(model_dir, 'net_awb.pth')
    
    if os.path.exists(awb_path):
        net_awb = load_model(awb_path)
        net_awb.to(device=device)
        net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                            map_location=device))
        net_awb.eval()
           
    elif os.path.exists(os.path.join(model_dir, 'net.pth')):
        net = load_model(os.path.join(model_dir, 'net.pth'))
        net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
        net_awb, _, _ = splitNetworks.splitNetworks(net)
        net_awb.to(device=device)
        net_awb.eval()
    else:
        raise Exception('Model not found!!')
    
    return net_awb
    

def white_balance_gui(input_image, net_awb, task='AWB',mxsize=656, device='cuda'):
        
    out_awb = deep_wb(input_image, task=task.lower(), net_awb=net_awb, device=device, s=mxsize)
            
    return (out_awb * 255).astype(np.uint8)
        
                    
######### Aux functions #########

def load_model(model_path):
    logging.info(f"Loading model {model_path}")
    
    # Choose models depends on which models we have
    if any(filename in model_path for filename in ['net_awb.pth', 'net_t.pth', 'net_s.pth']):
        model = deep_wb_single_task.deepWBnet()
    elif os.path.exists(model_path):
        model = deep_wb_model.deepWBNet()

    return model

def load_all_models(model_dir, device):
    awb_path = os.path.join(model_dir, 'net_awb.pth')
    t_path = os.path.join(model_dir, 'net_t.pth')
    s_path = os.path.join(model_dir, 'net_s.pth')
    
    if os.path.exists(awb_path) and os.path.exists(t_path) and os.path.exists(s_path):
        net_awb = load_model(awb_path)
        net_awb.to(device=device)
        net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                            map_location=device))
        net_awb.eval()
        
        net_t = load_model(t_path)
        net_t.to(device=device)
        net_t.load_state_dict(
            torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
        net_t.eval()
        
        net_s = load_model(s_path)
        net_s.to(device=device)
        net_s.load_state_dict(
            torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
        net_s.eval()
        
    elif os.path.exists(os.path.join(model_dir, 'net.pth')):
        net = load_model(os.path.join(model_dir, 'net.pth'))
        net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
        net_awb, net_t, net_s = splitNetworks.splitNetworks(net)
        net_awb.to(device=device)
        net_awb.eval()
        net_t.to(device=device)
        net_t.eval()
        net_s.to(device=device)
        net_s.eval()
        
    else:
        raise Exception('Model not found')

    return net_awb, net_t, net_s

def process_images(input_dir):
    
    imgfiles = []
    valid_images = (".jpg", ".png")
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(valid_images):
            imgfiles.append(os.path.join(input_dir, fn))
            
    return imgfiles

def all_filter(imgfiles,task, device, net_awb, net_s, net_t, S, tosave, show, out_dir):

    for fn in imgfiles:

        logging.info("Processing image {} ...".format(fn))
        img = Image.open(fn)
        _, fname = os.path.split(fn)
        name, _ = os.path.splitext(fname)
        
        out_awb, out_t, out_s = deep_wb(img, task=task.lower(), net_awb=net_awb, net_s=net_s, net_t=net_t,
                                        device=device, s=S)
        out_f, out_d, out_c = utils.colorTempInterpolate(out_t, out_s)
        if tosave:
            result_awb, result_t, result_f, result_d, result_c, result_s = save_all(name,out_awb, out_t, out_s, out_f, out_d, out_c, out_dir)
            
        if show:
            logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
            utils.imshow(img, result_awb, result_t, result_f, result_d, result_c, result_s)

def awb_filter(imgfiles,task, device, net_awb, S, tosave, show, out_dir): 
    
    for fn in imgfiles:

        logging.info("Processing image {} ...".format(fn))
        img = Image.open(fn)
        _, fname = os.path.split(fn)
               
        out_awb = deep_wb(img, task=task.lower(), net_awb=net_awb, device=device, s=S)
        if tosave:
            result_awb = utils.to_image(out_awb)
            result_awb.save(os.path.join(out_dir, fname))

        if show:
            logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
            utils.imshow(img, result_awb)

def edit_filter(imgfiles,task, device, net_s, net_t, S, tosave, show, target_color_temp, out_dir):        
    for fn in imgfiles:

            logging.info("Processing image {} ...".format(fn))
            img = Image.open(fn)
            _, fname = os.path.split(fn)
            name, _ = os.path.splitext(fname)
            out_t, out_s = deep_wb(img, task=task.lower(), net_s=net_s, net_t=net_t, device=device, s=S)

            if target_color_temp:
                out = utils.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
                if tosave:
                    out = utils.to_image(out)
                    out.save(os.path.join(out_dir, name + '_%d.png' % target_color_temp))

                if show:
                    logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
                    utils.imshow(img, out, colortemp=target_color_temp)

            else:
                out_f, out_d, out_c = utils.colorTempInterpolate(out_t, out_s)
                if tosave:
                    result_t, result_f, result_d, result_c, result_s = save_editing(name,out_t, out_s, out_f, out_d, out_c, out_dir)

                if show:
                    logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
                    utils.imshow(img, result_t, result_f, result_d, result_c, result_s)
                    
def save_all(name,out_awb, out_t, out_s, out_f, out_d, out_c, out_dir):
    result_awb = utils.to_image(out_awb)
    result_t = utils.to_image(out_t)
    result_s = utils.to_image(out_s)
    result_f = utils.to_image(out_f)
    result_d = utils.to_image(out_d)
    result_c = utils.to_image(out_c)
    result_awb.save(os.path.join(out_dir, name + '_AWB.png'))
    result_s.save(os.path.join(out_dir, name + '_S.png'))
    result_t.save(os.path.join(out_dir, name + '_T.png'))
    result_f.save(os.path.join(out_dir, name + '_F.png'))
    result_d.save(os.path.join(out_dir, name + '_D.png'))
    result_c.save(os.path.join(out_dir, name + '_C.png'))
    
    return result_awb, result_t, result_f, result_d, result_c, result_s

def save_editing(name,out_t, out_s, out_f, out_d, out_c, out_dir):
    result_t = utils.to_image(out_t)
    result_s = utils.to_image(out_s)
    result_f = utils.to_image(out_f)
    result_d = utils.to_image(out_d)
    result_c = utils.to_image(out_c)
    result_s.save(os.path.join(out_dir, name + '_S.png'))
    result_t.save(os.path.join(out_dir, name + '_T.png'))
    result_f.save(os.path.join(out_dir, name + '_F.png'))
    result_d.save(os.path.join(out_dir, name + '_D.png'))
    result_c.save(os.path.join(out_dir, name + '_C.png'))
    
    return result_t, result_f, result_d, result_c, result_s


####### MAIN #######
if __name__ == '__main__':
    model_dir = 'C:\\Users\\JoelVP\\Desktop\\UPV\\ImageEnhancementTFG\\src\\models\\Deep_White_Balance\\PyTorch\\models'

    input_dir = '.\\images'
    out_dir = '.\\mid_images'
    task = 'all'  # Cambiar a 'editing' o 'awb' para probar otras tareas.
    target_color_temp = None
    mxsize = 656
    show = False
    tosave = True
    device = 'cuda'

    white_balance(model_dir, input_dir, out_dir, task, target_color_temp, mxsize, show, tosave, device)