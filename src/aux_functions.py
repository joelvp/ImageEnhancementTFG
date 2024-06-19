import os
import tempfile
import random
from typing import List
import cv2
import logging
import numpy as np
import time
import gradio as gr

from models.utils import reset_gradio_flag
from src.objects.model_manager import ModelManager
from google_images_search import GoogleImagesSearch

import configparser
config = configparser.ConfigParser()
config.read('data\config.ini')


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_images_before_text(history, text_index):
    input_images = []
    images_found = False
    for next_item in reversed(history[:text_index]):
        if isinstance(next_item[0], tuple):
            input_images.append(next_item[0][0])
            images_found = True
        elif isinstance(next_item[0], str) and images_found:
            break
    return input_images, images_found


def extract_images_and_text(history):
    for item in reversed(history):
        if isinstance(item[0], str):
            text_index = history.index(item)
            input_images, images_found = find_images_before_text(history, text_index)
            return input_images, item[0] if images_found or not input_images else None
        elif isinstance(item[0], tuple):
            return [item[0][0]], None
        elif item[0] is None and item[1] is None:
            break
    return [], None


def handle_text_input(history, input_text, llama_model):
    response = llama_model.generate(input_text, False)
    return add_response(history, response.response_text)


def handle_image_input(history, llama_model):
    response = llama_model.generate(None, True)
    return add_response(history, response.response_text)


def handle_image_and_text_input(history, input_images, input_text, llama_model, model_manager):
    response = llama_model.generate(input_text, True)
    if response.tasks:
        if response.new_background_image:
            response_message = f"El cielo se va a reemplazar por {response.new_background_image}"
            for updated_history in add_response(history, response_message):
                yield updated_history
            sky = google_image_search(response.new_background_image)
            enhanced_images = apply_transformations(input_images, ['Sky'], model_manager, sky_image=sky)
        elif 'No Tasks' in response.tasks:
            enhanced_images = []
            for updated_history in add_response(history, response.response_text):
                yield updated_history
        else:
            task_list = ", ".join(response.tasks)
            response_message = f"La imagen se va a mejorar con {task_list}"
            for updated_history in add_response(history, response_message):
                yield updated_history
            enhanced_images = apply_transformations(input_images, response.tasks, model_manager)

        output_img_paths = images_to_temp_paths(enhanced_images)
        for img_path in output_img_paths:
            history.append((None, (img_path,)))
            yield history

    elif response.error_message:
        for updated_history in add_response(history, response.error_message):
            yield updated_history
    else:
        generic_message = 'Especifica mejor la tarea que quieres aplicar a la imagen.'
        for updated_history in add_response(history, generic_message):
            yield updated_history


def process_message(history, message):
    if not message["files"] and message["text"] == '':
        history.append((None, None))
    else:
        for x in message["files"]:
            history.append(((x,), None))
        if message["text"] != '':
            history.append((message["text"], None))
    return history


def add_response(history, response):
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history

  
def apply_transformations(input_images, options, model_manager: ModelManager, sky_image=None) -> List[np.ndarray]:
    enhanced_images = []

    for image in input_images:
        # Numpy array
        image = imread(image)

        for option in options:
            try:
                if option == "Low Light":
                    if model_manager.ll_model is None:
                        model_manager.load_ll_model()

                    logging.info("Applying Low Light")
                    image = model_manager.ll_model.process_image(image)
                    logging.info("Low Light applied!")

                elif option == "Denoise":
                    if model_manager.denoise_model is None:
                        model_manager.load_denoise_model()

                    logging.info("Applying Denoising")
                    image = model_manager.denoise_model.process_image(image)
                    logging.info("Denoising applied!")

                elif option == "Deblur":
                    if model_manager.deblur_model is None:
                        model_manager.load_deblur_model()

                    logging.info("Applying Deblurring")
                    image = model_manager.deblur_model.process_image(image)
                    logging.info("Deblurring applied!")

                elif option == "White Balance":
                    if model_manager.wb_model is None:
                        model_manager.load_wb_model()

                    logging.info("Applying White Balance")
                    image = model_manager.wb_model.process_image(image)
                    logging.info("White Balance applied!")

                elif option == "Sky":
                    if sky_image is None:
                        raise gr.Error("Sky image needed!")

                    if model_manager.sky_model is None:
                        model_manager.load_sky_model()

                    logging.info("Applying Sky Replacement")
                    image = model_manager.sky_model.process_image(image, sky_image)
                    logging.info("Sky Replacement applied!")

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    reset_gradio_flag()
                    logging.error(f"Error applying {option}: CUDA error out of memory")
                    gr.Info(f"Not applied {option} because CUDA error out of memory")

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
    API_KEY = config['api_keys']['google_search']
    CX = config['google_search']['cx']
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

            # Leer la imagen usando imread
            image = imread(selected_image_path)
            return image
        else:
            print("No se pudo descargar ninguna imagen.")
            return None