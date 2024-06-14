import os
import torch
import numpy as np
from src.objects.model import Model
from models.White_Balance.arch import deep_wb_model, deep_wb_single_task, splitNetworks
from models.White_Balance.utilities.deepWB import deep_wb


class WhiteBalance(Model):
    def __init__(self, model_path='./models/White_Balance/models', device='cuda'):
        super().__init__(model_path, device)

    def load_model(self):
        # Load only AWB model
        awb_path = os.path.join(self.model_path, 'net_awb.pth')
        if os.path.exists(awb_path):
            net_awb = self._load_awb_model(awb_path)
        elif os.path.exists(os.path.join(self.model_path, 'net.pth')):
            net_awb = self._load_split_model(os.path.join(self.model_path, 'net.pth'))
        else:
            raise Exception('Model not found!!')
        self.model = net_awb

    def process_image(self, input_image, task='AWB', maxsize=656):
        out_awb = deep_wb(input_image, task=task.lower(), net_awb=self.model, device=self.device, s=maxsize)
        return (out_awb * 255).astype(np.uint8)

    def _load_awb_model(self, awb_path):
        net_awb = self._get_model_instance(awb_path)
        net_awb.to(device=self.device)
        net_awb.load_state_dict(torch.load(awb_path, map_location=self.device))
        net_awb.eval()
        return net_awb

    def _load_split_model(self, net_path):
        net = self._get_model_instance(net_path)
        net.load_state_dict(torch.load(net_path))
        net_awb, _, _ = splitNetworks.splitNetworks(net)
        net_awb.to(device=self.device)
        net_awb.eval()
        return net_awb

    def _get_model_instance(self, model_path):
        # Si es uno de los modelos especificos cargas un tipo de arquitectura, si es el net.pth cargas el modelo completo
        if any(filename in model_path for filename in ['net_awb.pth', 'net_t.pth', 'net_s.pth']):
            return deep_wb_single_task.deepWBnet()
        else:
            return deep_wb_model.deepWBNet()


