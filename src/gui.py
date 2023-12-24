import gradio as gr
import numpy as np
import logging
import cv2
import os

from models.Deep_White_Balance.PyTorch.white_balance import load_wb_model, white_balance_gui
from models.LLFlow.code.lowlight import load_ll_model, lowlight_gui
from models.NAFNet.denoise import load_denoising_model, denoising_gui
from models.SkyAR.sky_replace import load_sky_model, sky_replace_gui

logging.basicConfig(level=logging.INFO)


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def apply_transformations(input_images, options, sky_image=None):

    enhanced_images = []

    # Flags for load models only one time
    wb_flag = False
    ll_flag = False
    denoise_flag = False
    sky_flag = False

    for image in input_images:
        # Numpy array
        image = imread(image)

        for option in options:
            if option == "Low Light":
                if not ll_flag:
                    logging.info("Loading Low Light model")
                    model, opt = load_ll_model()
                    ll_flag = True

                logging.info("Applying Low Light")
                image = lowlight_gui(image, model, opt)

            elif option == "Denoise":
                if not denoise_flag:
                    logging.info("Loading Denoising model")
                    model = load_denoising_model()
                    denoise_flag = True

                logging.info("Applying Denoising")
                image = denoising_gui(image, model)

            elif option == "White Balance":
                if not wb_flag:
                    logging.info('Loading AWB model')
                    net_awb = load_wb_model()
                    wb_flag = True

                logging.info("Applying White Balance")
                image = white_balance_gui(image, net_awb)
                
            elif option == "Sky":
                if sky_image is None:
                    raise gr.Error("Sky image needed!")
                
                if not sky_flag:
                    logging.info('Loading Sky Replacement model')
                    model, config = load_sky_model()
                    sky_flag = True
                    
                logging.info("Applying Sky Replacement")
                image = sky_replace_gui(image, sky_image, model, config)

        enhanced_images.append(image)

    return enhanced_images

def update_images(input_images): return input_images

def switch(options: list):
    """ Sky image input visibility""" 
    if "Sky" in options:
        return gr.Image(visible=True)
        
    else:
        return gr.Image(visible=False)

with gr.Blocks() as demo:
    input_images = gr.File(type="filepath", label="Input Images", file_count="multiple", file_types=["image"], interactive=True)

    input_gallery = gr.Gallery(
        label="Input Images",
        elem_id="input_gallery",
        show_label=False,
        columns=[3],
        rows=[1],
        object_fit="contain",
        height="auto"
    )

    input_images.change(update_images, [input_images], input_gallery)

    options = gr.Dropdown(
            ["Low Light", "Denoise", "White Balance", "Sky"], value=["White Balance"], multiselect=True,
            label="Choose the filters according to the order in which you want to apply them. "
        )

    # Sky image input (initially hidden)
    sky_image_input = gr.Image(
    "../data/demo_images/skybox/galaxy.jpg",
    elem_id="sky_image_input",
    label="Sky Image",
    visible=False,
    height=500,  # Establece la altura máxima de la imagen
    width=10000    # Establece la anchura máxima de la imagen
    )
    
    # Sky image input depends on Sky CheckBox
    options.change(switch, options, sky_image_input)

    btn_submit = gr.Button("Submit", scale=0)

    output_gallery = gr.Gallery(
        label="Output Images",
        elem_id="output_gallery",
        show_label=False,
        columns=[3],
        rows=[1],
        object_fit="contain",
        height="auto"
    )

    btn_submit.click(apply_transformations, [input_images, options, sky_image_input], output_gallery)

if __name__ == "__main__":
    demo.launch()
    
    
##### HOW TO RUN #####
    """
    cd src
    python gui.py
    
    """