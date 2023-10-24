import os
import logging
from PIL import Image
import torch

from arch import deep_wb_model, deep_wb_single_task
from utilities import utils, deepWB


def white_balance(model_dir='./models/Deep_White_Balance/PyTorch/models',input_dir='./images/',out_dir='./mid_images',
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
        print(model_dir)
        net_awb, net_t, net_s = load_all_models(model_dir, device)
        logging.info("Models loaded !")
        
    elif task.lower() == 'editing':
        t_path = os.path.join(model_dir, 'net_t.pth')
        s_path = os.path.join(model_dir, 'net_s.pth')

        if os.path.exists(t_path) and os.path.exists(s_path):
            net_t = load_model(t_path, device)
            net_s = load_model(s_path, device)
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = load_model(os.path.join(model_dir, 'net.pth'), device)
            _, net_t, net_s = deep_wb_model.splitter.splitNetworks(net)
        else:
            raise Exception('Model not found!')
        
    elif task.lower() == 'awb':
        awb_path = os.path.join(model_dir, 'net_awb.pth')
        
        if os.path.exists(awb_path):
            net_awb = load_model(awb_path, device)
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = load_model(os.path.join(model_dir, 'net.pth'), device)
            net_awb, _, _ = deep_wb_model.splitter.splitNetworks(net)
        else:
            raise Exception('Model not found!')
    else:
        raise Exception("Wrong task! Task should be: 'AWB', 'editing', or 'all'")


    # imgfiles = []
    # valid_images = (".jpg", ".png")
    # for fn in os.listdir(input_dir):
    #     if fn.lower().endswith(valid_images):
    #         imgfiles.append(os.path.join(input_dir, fn))

    # for fn in imgfiles:

    #     logging.info("Processing image {} ...".format(fn))
    #     img = Image.open(fn)
    #     _, fname = os.path.split(fn)
    #     print(fname)
    #     name, _ = os.path.splitext(fname)
    #     if task.lower() == 'all':  # awb and editing tasks
    #         out_awb, out_t, out_s = deep_wb(img, task=task.lower(), net_awb=net_awb, net_s=net_s, net_t=net_t,
    #                                         device=device, s=S)
    #         out_f, out_d, out_c = utils.colorTempInterpolate(out_t, out_s)
    #         if tosave:
    #             result_awb = utils.to_image(out_awb)
    #             result_t = utils.to_image(out_t)
    #             result_s = utils.to_image(out_s)
    #             result_f = utils.to_image(out_f)
    #             result_d = utils.to_image(out_d)
    #             result_c = utils.to_image(out_c)
    #             result_awb.save(os.path.join(out_dir, name + '_AWB.png'))
    #             result_s.save(os.path.join(out_dir, name + '_S.png'))
    #             result_t.save(os.path.join(out_dir, name + '_T.png'))
    #             result_f.save(os.path.join(out_dir, name + '_F.png'))
    #             result_d.save(os.path.join(out_dir, name + '_D.png'))
    #             result_c.save(os.path.join(out_dir, name + '_C.png'))

    #         if show:
    #             logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
    #             utils.imshow(img, result_awb, result_t, result_f, result_d, result_c, result_s)

    #     elif task.lower() == 'awb':  # awb task
    #         out_awb = deep_wb(img, task=task.lower(), net_awb=net_awb, device=device, s=S)
    #         if tosave:
    #             result_awb = utils.to_image(out_awb)
    #             result_awb.save(os.path.join(out_dir, fname))

    #         if show:
    #             logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
    #             utils.imshow(img, result_awb)

    #     else:  # editing
    #         out_t, out_s = deep_wb(img, task=task.lower(), net_s=net_s, net_t=net_t, device=device, s=S)

    #         if target_color_temp:
    #             out = utils.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
    #             if tosave:
    #                 out = utils.to_image(out)
    #                 out.save(os.path.join(out_dir, name + '_%d.png' % target_color_temp))

    #             if show:
    #                 logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
    #                 utils.imshow(img, out, colortemp=target_color_temp)

    #         else:
    #             out_f, out_d, out_c = utils.colorTempInterpolate(out_t, out_s)
    #             if tosave:
    #                 result_t = utils.to_image(out_t)
    #                 result_s = utils.to_image(out_s)
    #                 result_f = utils.to_image(out_f)
    #                 result_d = utils.to_image(out_d)
    #                 result_c = utils.to_image(out_c)
    #                 result_s.save(os.path.join(out_dir, name + '_S.png'))
    #                 result_t.save(os.path.join(out_dir, name + '_T.png'))
    #                 result_f.save(os.path.join(out_dir, name + '_F.png'))
    #                 result_d.save(os.path.join(out_dir, name + '_D.png'))
    #                 result_c.save(os.path.join(out_dir, name + '_C.png'))

    #             if show:
    #                 logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
    #                 utils.imshow(img, result_t, result_f, result_d, result_c, result_s)
                    
######### Aux functions #########

def load_model(model_path, device):
    logging.info(f"Loading model {model_path}")
    
    if any(os.path.exists(path) for path in [model_path, 'net_awb.pth', 'net_t.pth', 'net_s.pth']):
        model = deep_wb_single_task.deepWBnet()
    elif os.path.exists(model_path):
        model = deep_wb_model.deepWBNet()

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_all_models(model_dir, device):
    awb_path = os.path.join(model_dir, 'net_awb.pth')
    t_path = os.path.join(model_dir, 'net_t.pth')
    s_path = os.path.join(model_dir, 'net_s.pth')
    
    #TODO: Revisar porque no entra en el primer if cuando las rutas estan bien 
    print(awb_path,t_path,s_path)

    if os.path.exists(awb_path) and os.path.exists(t_path) and os.path.exists(s_path):
        net_awb = load_model(awb_path, device)
        net_t = load_model(t_path, device)
        net_s = load_model(s_path, device)
    elif os.path.exists(os.path.join(model_dir, 'net.pth')):
        net = load_model(os.path.join(model_dir, 'net.pth'), device)
        net_awb, net_t, net_s = deep_wb_model.splitter.splitNetworks(net)
    else:
        raise Exception('Model not found')

    return net_awb, net_t, net_s

####### MAIN #######
if __name__ == '__main__':
    model_dir = '.\\models'
    input_dir = '.\\images'
    out_dir = '.\\mid_images'
    task = 'all'  # Cambiar a 'editing' o 'awb' para probar otras tareas.
    target_color_temp = None
    mxsize = 656
    show = False
    tosave = True
    device = 'cuda'

    white_balance(model_dir, input_dir, out_dir, task, target_color_temp, mxsize, show, tosave, device)