from models.LLFlow.code.test_unpaired import *

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