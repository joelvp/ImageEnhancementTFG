import os
import logging
import torch
import numpy as np

from .arch import deep_wb_model, deep_wb_single_task, splitNetworks
from .utilities.deepWB import deep_wb

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
