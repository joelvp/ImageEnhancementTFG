import os
import tempfile
from io import BytesIO
import random
from typing import List
import cv2
import logging
import gradio as gr
import numpy as np
import base64

from PIL import Image

from src.objects.model_manager import ModelManager
from google_images_search import GoogleImagesSearch
from models.LLFlow.code.lowlight import lowlight_gui
from models.SkyAR.sky_replace import sky_replace_gui


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Image to Base 64 Converter
def image_path_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')


def image_to_base64(image_array):
    # Convertir el array de NumPy a una imagen en formato PIL
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

    # Crear un buffer de bytes
    buffered = BytesIO()

    # Guardar la imagen en el buffer en formato JPEG
    image.save(buffered, format="JPEG")

    # Obtener los datos binarios de la imagen
    img_bytes = buffered.getvalue()

    # Codificar los datos binarios en base64
    encoded_string = base64.b64encode(img_bytes)

    return encoded_string.decode('utf-8')

  
# Function that takes User Inputs and displays it on ChatUI
def query_message(history, txt, img):
    if not img:
        history += [(txt, None)]
        return history
    base64_img = image_path_to_base64(img[0]) # Temporal solo pa una imagen
    data_url = f"data:image/jpeg;base64,{base64_img}"
    history += [(f"{txt} ![]({data_url})", None)]
    return history

  
def apply_transformations(input_images, options, model_manager: ModelManager, sky_image=None) -> List[np.ndarray]:
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
                image = model_manager.denoise_model.process_image(image)
                
            elif option == "Deblur":
                if model_manager.deblur_model is None:
                    model_manager.load_deblur_model()

                logging.info("Applying Deblurring")
                image = model_manager.deblur_model.process_image(image)

            elif option == "White Balance":
                if model_manager.wb_model is None:
                    model_manager.load_wb_model()

                logging.info("Applying White Balance")
                image = model_manager.wb_model.process_image(image)

            elif option == "Sky":
                if sky_image is None:
                    raise gr.Error("Sky image needed!")

                if model_manager.sky_model is None:
                    model_manager.load_sky_model()

                logging.info("Applying Sky Replacement")
                image = sky_replace_gui(image, sky_image, model_manager.sky_model, model_manager.sky_config)

        enhanced_images.append(image)
        
    logging.info("Images enhanced correctly!")

    return enhanced_images


def images_to_temp_paths(images: List[np.ndarray]) -> List[str]:
    temp_paths = []
    for image in images:
        # Crear un archivo temporal con extensión .png
        temp_file_path = tempfile.mktemp(suffix=".jpg")
        temp_paths.append(temp_file_path)
        # Guardar la imagen en el archivo temporal
        cv2.imwrite(temp_file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return temp_paths


def google_image_search(query: str):
    # Configura tu API key y tu motor de búsqueda
    API_KEY = 'AIzaSyCHGH5e0AKtnchJHOZLHgtlMzfG0Zb2YU8'
    CX = '46ef2a835eabd4047'
    search_params = {
        'q': query,
        'num': 5,  # Buscar 5 imágenes en lugar de 1
        'fileType': 'jpg|png',
        'safe': 'off',
        'imgType': 'photo',
        'imgSize': 'large'
    }
    # Inicializa el objeto GoogleImagesSearch
    gis = GoogleImagesSearch(API_KEY, CX)
    # Crear un directorio temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        # Realiza una búsqueda de imágenes con la palabra clave
        gis.search(search_params=search_params, path_to_dir=temp_dir)

        # Obtener las rutas de las imágenes guardadas en el directorio temporal
        downloaded_image_paths = []
        if gis.results():
            for image_result in gis.results():
                filename = os.path.basename(image_result.path)
                downloaded_image_path = os.path.join(temp_dir, filename)
                downloaded_image_paths.append(downloaded_image_path)

        if downloaded_image_paths:
            # Seleccionar una imagen aleatoria de las descargadas
            selected_image_path = random.choice(downloaded_image_paths)
            print(f'Selected image: {selected_image_path}')

            # Leer la imagen usando imread
            image = imread(selected_image_path)
            return image
        else:
            print("No se pudo descargar ninguna imagen.")
            return None