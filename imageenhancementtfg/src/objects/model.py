import torch
import logging
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt

from models.utils import clear_cuda_cache, reset_gradio_flag


class Model(ABC):
    def __init__(self, model_path, config_path, device='cuda'):
        self.device = torch.device('cuda' if device.lower() == 'cuda' and torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        self.model_path = model_path
        self.config_path = config_path
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @retry(stop=stop_after_attempt(3), before=clear_cuda_cache, reraise=True)
    def process_image(self, input_image):
        image = self._process_image_impl(input_image)
        reset_gradio_flag()
        return image

    @abstractmethod
    def _process_image_impl(self, input_image):
        pass
