import torch
import logging
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device('cuda' if device.lower() == 'cuda' and torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def process_image(self, input_image):
        pass
