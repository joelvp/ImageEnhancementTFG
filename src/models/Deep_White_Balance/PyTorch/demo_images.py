

def white_balance(model_dir='./models/Deep_White_Balance/PyTorch/models',input_dir='./images/',out_dir='./mid_images',
                  task='AWB',target_color_temp=None,
                  mxsize=656,show=False,tosave=True,device='cuda'):
    S = mxsize

    if device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if target_color_temp:
        assert 2850 <= target_color_temp <= 7500, (
                'Color temperature should be in the range [2850 - 7500], but the given one is %d' % target_color_temp)

        if task.lower() != 'editing':
            raise Exception('The task should be editing when a target color temperature is specified.')

    logging.info(f'Using device {device}')

    if tosave:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    if task.lower() == 'all':
        if os.path.exists(os.path.join(model_dir, 'net_awb.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_s.pth')):
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_awb.pth')))
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_t.pth')))
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_s.pth')))
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
            logging.info("Models loaded !")
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net.pth')))
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, net_t, net_s = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
    elif task.lower() == 'editing':
        if os.path.exists(os.path.join(model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_s.pth')):
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_t.pth')))
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_s.pth')))
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
            logging.info("Models loaded !")
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net.pth')))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            _, net_t, net_s = splitter.splitNetworks(net)
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
    elif task.lower() == 'awb':
        if os.path.exists(os.path.join(model_dir, 'net_awb.pth')):
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net_awb.pth')))
            logging.info(f'Using device {device}')
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            logging.info("Loading model {}".format(os.path.join(model_dir, 'net.pth')))
            logging.info(f'Using device {device}')
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, _, _ = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
        else:
            raise Exception('Model not found!')
    else:
        raise Exception("Wrong task! Task should be: 'AWB', 'editing', or 'all'")


    imgfiles = []
    valid_images = (".jpg", ".png")
    for fn in os.listdir(input_dir):
        if fn.lower().endswith(valid_images):
            imgfiles.append(os.path.join(input_dir, fn))

    for fn in imgfiles:

        logging.info("Processing image {} ...".format(fn))
        img = Image.open(fn)
        _, fname = os.path.split(fn)
        name, _ = os.path.splitext(fname)
        if task.lower() == 'all':  # awb and editing tasks
            out_awb, out_t, out_s = deep_wb(img, task=task.lower(), net_awb=net_awb, net_s=net_s, net_t=net_t,
                                            device=device, s=S)
            out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
            if tosave:
                result_awb = utls.to_image(out_awb)
                result_t = utls.to_image(out_t)
                result_s = utls.to_image(out_s)
                result_f = utls.to_image(out_f)
                result_d = utls.to_image(out_d)
                result_c = utls.to_image(out_c)
                result_awb.save(os.path.join(out_dir, name + '_AWB.png'))
                result_s.save(os.path.join(out_dir, name + '_S.png'))
                result_t.save(os.path.join(out_dir, name + '_T.png'))
                result_f.save(os.path.join(out_dir, name + '_F.png'))
                result_d.save(os.path.join(out_dir, name + '_D.png'))
                result_c.save(os.path.join(out_dir, name + '_C.png'))

            if show:
                logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
                utls.imshow(img, result_awb, result_t, result_f, result_d, result_c, result_s)

        elif task.lower() == 'awb':  # awb task
            out_awb = deep_wb(img, task=task.lower(), net_awb=net_awb, device=device, s=S)
            if tosave:
                result_awb = utls.to_image(out_awb)
                result_awb.save(os.path.join(out_dir, name + '_AWB.png'))

            if show:
                logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
                utls.imshow(img, result_awb)

        else:  # editing
            out_t, out_s = deep_wb(img, task=task.lower(), net_s=net_s, net_t=net_t, device=device, s=S)

            if target_color_temp:
                out = utls.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
                if tosave:
                    out = utls.to_image(out)
                    out.save(os.path.join(out_dir, name + '_%d.png' % target_color_temp))

                if show:
                    logging.info("Visualizing result for image: {}, close to continue ...".format(fn))
                    utls.imshow(img, out, colortemp=target_color_temp)

            else:
                out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
                if tosave:
                    result_t = utls.to_image(out_t)
                    result_s = utls.to_image(out_s)
                    result_f = utls.to_image(out_f)
                    result_d = utls.to_image(out_d)
                    result_c = utls.to_image(out_c)
                    result_s.save(os.path.join(out_dir, name + '_S.png'))
                    result_t.save(os.path.join(out_dir, name + '_T.png'))
                    result_f.save(os.path.join(out_dir, name + '_F.png'))
                    result_d.save(os.path.join(out_dir, name + '_D.png'))
                    result_c.save(os.path.join(out_dir, name + '_C.png'))

                if show:
                    logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
                    utls.imshow(img, result_t, result_f, result_d, result_c, result_s)

