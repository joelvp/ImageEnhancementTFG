from models.LLFlow.code.test_unpaired import *
from PIL import Image

def lowlight(conf_path="./models/LLFlow/code/confs/LOLv2-pc.yml",out_dir="../data/demo_images/output/"):

    model, opt = load_model(conf_path)
    # model.netG = model.netG.to("cpu")
    model.netG = model.netG.cuda()

    lr_dir = opt['dataroot_unpaired']
    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.*'))

    for lr_path, idx_test in tqdm.tqdm(zip(lr_paths, range(len(lr_paths)))):

        lr = imread(lr_path)
        raw_shape = lr.shape
        lr, padding_params = auto_padding(lr)
        his = hiseq_color_cv2_img(lr)
        if opt.get("histeq_as_input", False):
            lr = his

        lr_t = t(lr)
        if opt["datasets"]["train"].get("log_low", False):
            lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
        if opt.get("concat_histeq", False):
            his = t(his)
            lr_t = torch.cat([lr_t, his], dim=1)
        with torch.cuda.amp.autocast():
            # sr_t = model.get_sr(lq=lr_t, heat=None)
            sr_t = model.get_sr(lq=lr_t.cuda(), heat=None)

        sr = rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
                 padding_params[2]:sr_t.shape[3] - padding_params[3]])
        assert raw_shape == sr.shape
        path_out_sr = os.path.join(out_dir, os.path.basename(lr_path))
        imwrite(path_out_sr, sr)
        
def load_ll_model(conf_path="./models/LLFlow/code/confs/LOLv2-pc.yml"):
    model, opt = load_model(conf_path)
    model.netG = model.netG.cuda()
    
    return model, opt
    
         
def lowlight_gui(input_image, model, opt):
    
    # Convertir la imagen de Pillow a una matriz NumPy
    image = np.array(input_image)
    # If the image is in floating-point format, scale it to the range [0, 255]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    raw_shape = image.shape
    image, padding_params = auto_padding(image)
    his = hiseq_color_cv2_img(image)
    lr_t = t(image)
    if opt["datasets"]["train"].get("log_low", False):
        lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
    if opt.get("concat_histeq", False):
        his = t(his)
        lr_t = torch.cat([lr_t, his], dim=1)
    with torch.cuda.amp.autocast():
        sr_t = model.get_sr(lq=lr_t.cuda(), heat=None)

    sr = rgb(torch.clamp(sr_t, 0, 1)[:, :, padding_params[0]:sr_t.shape[2] - padding_params[1],
                padding_params[2]:sr_t.shape[3] - padding_params[3]])
    assert raw_shape == sr.shape
    
    return Image.fromarray(sr.astype('uint8')) # Return Pillow image

    
    