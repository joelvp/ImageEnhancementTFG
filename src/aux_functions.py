from typing import List
import cv2
import logging
import gradio as gr
import numpy as np
import base64

from models.NAFNet.deblur import deblurring_gui

from .model_manager import ModelManager

from models.Deep_White_Balance.PyTorch.white_balance import white_balance_gui
from models.LLFlow.code.lowlight import  lowlight_gui
from models.NAFNet.denoise import  denoising_gui
from models.SkyAR.sky_replace import  sky_replace_gui


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Image to Base 64 Converter
def image_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

# Function that takes User Inputs and displays it on ChatUI
def query_message(history,txt,img):
    if not img:
        history += [(txt,None)]
        return history
    base64_img = image_to_base64(img)
    data_url = f"data:image/jpeg;base64,{base64_img}"
    history += [(f"{txt} ![]({data_url})", None)]
    return history

def apply_transformations(input_images, options, model_manager:ModelManager, sky_image=None) -> List[np.ndarray]:
    enhanced_images = []

    for image in input_images:
        # Numpy array
        image = imread(image)

        for option in options:
            if option == "Low Light":
                if model_manager.ll_model is None:
                    model_manager.load_ll_model()

                logging.info("Applying Low Light")
                image = lowlight_gui(image, model_manager.ll_model, model_manager.opt_ll)

            elif option == "Denoise":
                if model_manager.denoise_model is None:
                    model_manager.load_denoise_model()

                logging.info("Applying Denoising")
                image = denoising_gui(image, model_manager.denoise_model)
                
            elif option == "Deblur":
                if model_manager.deblur_model is None:
                    model_manager.load_deblur_model()

                logging.info("Applying Deblurring")
                image = deblurring_gui(image, model_manager.deblur_model)


            elif option == "White Balance":
                if model_manager.wb_model is None:
                    model_manager.load_wb_model()

                logging.info("Applying White Balance")
                image = white_balance_gui(image, model_manager.wb_model)

            elif option == "Sky":
                if sky_image is None:
                    raise gr.Error("Sky image needed!")

                if model_manager.sky_model is None:
                    model_manager.load_sky_model()

                logging.info("Applying Sky Replacement")
                image = sky_replace_gui(image, sky_image, model_manager.sky_model, model_manager.sky_config)

        enhanced_images.append(image)
        
    logging.info("Filters correctly apllied!")

    return enhanced_images