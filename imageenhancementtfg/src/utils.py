import os
import tempfile
import random
from typing import List, Tuple, Optional, Generator, TypeAlias, Iterator, Union, Dict
import cv2
import logging
import numpy as np
import time
import gradio as gr

from models.Llama.llama import Llama
from models.utils import reset_gradio_flag, load_config
from src.objects.model_manager import ModelManager
from google_images_search import GoogleImagesSearch

config = load_config('data/config.ini')

HistoryItem: TypeAlias = Union[Tuple[str, str], Tuple[Optional[str], str], Tuple[str, Optional[str]], Tuple[Optional[Tuple[str]], Optional[str]]]
History: TypeAlias = List[HistoryItem]


def imread(img_path: str) -> np.ndarray:
    """
    Read an image from file and convert it to RGB format.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Image data as a NumPy array in RGB format.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_images_before_text(history: History, text_index: int) -> Tuple[List[str], bool]:
    """
    Find images before the text in the chat history.

    Parameters:
        history (History): Chat history.
        text_index (int): Index of the text in the history.

    Returns:
        Tuple[List[str], bool]: List of image paths and a boolean indicating if images were found.
    """
    input_images = []
    images_found = False
    for next_item in reversed(history[:text_index]):
        if isinstance(next_item[0], tuple):
            input_images.append(next_item[0][0])
            images_found = True
        elif isinstance(next_item[0], str) and images_found:
            break
    return input_images, images_found


def extract_images_and_text(history: History) -> Tuple[List[str], Optional[str]]:
    """
    Extract images and text from the chat history.

    Parameters:
        history (History): Chat history.

    Returns:
        Tuple[List[str], Optional[str]]: List of image paths and the input text.
    """
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


def handle_text_input(history: History, input_text: str, llama_model: Llama) -> Iterator[History]:
    """
    Handle text input for the LLM.

    Parameters:
        history (History): Chat history.
        input_text (str): Input text.
        llama_model (Llama): Llama model instance.

    Returns:
        Generator[Iterator]: Updated history with the response.
    """
    response = llama_model.generate(input_text, False)
    logging.info(f"LLM Response: {response}")
    return add_response(history, response.response_text)


def handle_image_input(history: History, llama_model: Llama) -> Iterator[History]:
    """
    Handle image input for the LLM.

    Parameters:
        history (History): Chat history.
        llama_model (Llama): Llama model instance.

    Returns:
        Iterator[History]: Updated history with the response.
    """
    response = llama_model.generate(None, True)
    logging.info(f"LLM Response: {response}")
    return add_response(history, response.response_text)


def handle_image_and_text_input(history: History, input_images: List[str], input_text: str, llama_model: Llama, model_manager: ModelManager) -> Iterator[History]:
    """
    Handle image and text input for the LLM.

    Parameters:
        history (History): Chat history.
        input_images (List[str]): List of input image paths.
        input_text (str): Input text.
        llama_model (Llama): Llama model instance.
        model_manager (ModelManager): Model manager instance.

    Returns:
        Iterator[History]: Updated history with the response and enhanced images.
    """
    response = llama_model.generate(input_text, True)
    logging.info(f"LLM Response: {response}")

    if response.tasks:
        if response.new_background_image:
            task_list = format_tasks_list(response.tasks)
            response_message = f"La imagen se va a mejorar {task_list}"
            for updated_history in add_response(history, response_message):
                yield updated_history
            sky = google_image_search(response.new_background_image)
            tasks = translate_strings(response.tasks)
            enhanced_images = apply_transformations(input_images, tasks, model_manager, sky_image=sky)

        elif 'No Tasks' in response.tasks:
            enhanced_images = []
            for updated_history in add_response(history, response.response_text):
                yield updated_history

        else:
            task_list = format_tasks_list(response.tasks)
            response_message = f"La imagen se va a mejorar {task_list}"
            for updated_history in add_response(history, response_message):
                yield updated_history
            tasks = translate_strings(response.tasks)
            enhanced_images = apply_transformations(input_images, tasks, model_manager)

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


def translate_strings(input_strings: List[str]) -> List[str]:
    """
    Translate a list of strings based on a predefined mapping.

    Parameters:
        input_strings (List[str]): List of strings to be translated.

    Returns:
        List[str]: Translated list of strings where applicable translations are applied.
    """
    translation_map = {
        "quitando el ruido": "Denoise",
        "enfocando": "Deblur",
        "ajustando el balance de blancos": "White Balance",
        "aumentando la luz": "Low Light",
        "corrigiendo la distorsion de lente": "Fish Eye",
        "cambiando el cielo": "Sky"

    }

    translated_list = []
    for string in input_strings:
        translated_string = translation_map.get(string, string)
        translated_list.append(translated_string)

    return translated_list


def format_tasks_list(items: List[str]) -> str:
    """
    Join a list of strings with commas and an 'and' before the last item.

    Parameters:
        items (List[str]): List of strings to be joined.

    Returns:
        str: A single string with items joined by commas and an 'and' before the last item.
    """
    if not items:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return " y ".join(items)
    else:
        return ", ".join(items[:-1]) + " y " + items[-1]


def process_message(history: History, message: Dict) -> History:
    """
    Process the user message and update the chat history.

    Parameters:
        history (History): Chat history.
        message (Dict): User message.

    Returns:
        History: "text" and "files", both optional, of the user input
    """
    if not message["files"] and message["text"] == '':
        history.append((None, None))
    else:
        for x in message["files"]:
            history.append(((x,), None))
        if message["text"] != '':
            history.append((message["text"], None))
    return history


def add_response(history: History, response: str) -> Iterator[History]:
    """
    Add a response to the chat history.

    Parameters:
        history (History): Chat history.
        response (str): Response text.

    Returns:
        Iterator[History]: Updated chat history with the response.
    """
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history

  
def apply_transformations(input_images: List[str], options: List[str], model_manager: ModelManager, sky_image: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """
    Apply transformations to input images based on selected options.

    Parameters:
        input_images (List[str]): List of input image paths.
        options (List[str]): List of selected options.
        model_manager (ModelManager): Model manager instance.
        sky_image (Optional[np.ndarray]): Sky image for replacement. Defaults to None.

    Returns:
        List[np.ndarray]: List of enhanced images as NumPy arrays.
    """
    enhanced_images = []

    for image in input_images:
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

                elif option == "Fish Eye":
                    if model_manager.lens_distortion_model is None:
                        model_manager.load_lens_distortion_model()

                    logging.info("Applying Fish Eye Correction")
                    image = model_manager.lens_distortion_model.process_image(image)
                    logging.info("Fish Eye correction applied!")

                else:
                    logging.warning(f"No valid option selected for enhancement: {option}")

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    reset_gradio_flag()
                    logging.error(f"Error applying {option}: CUDA error out of memory")
                    gr.Info(f"Not applied {option} because CUDA error out of memory")
                    continue

        enhanced_images.append(image)
        
    logging.info("Images enhanced correctly!")

    return enhanced_images


def images_to_temp_paths(images: List[np.ndarray]) -> List[str]:
    """
    Save enhanced images to temporary file paths.

    Parameters:
        images (List[np.ndarray]): List of images as NumPy arrays.

    Returns:
        List[str]: List of temporary file paths for the enhanced images.
    """
    temp_paths = []
    for image in images:
        temp_file_path = tempfile.mktemp(suffix=".jpg")
        temp_paths.append(temp_file_path)
        cv2.imwrite(temp_file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return temp_paths


def google_image_search(query: str) -> Optional[np.ndarray]:
    """
    Perform a Google image search and return the first image result.

    Parameters:
        query (str): Search query.

    Returns:
        Optional[np.ndarray]: Image data as a NumPy array or None if no image is found.
    """
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

    gis = GoogleImagesSearch(API_KEY, CX)
    with tempfile.TemporaryDirectory() as temp_dir:
        gis.search(search_params=search_params, path_to_dir=temp_dir)

        downloaded_image_paths = []
        if gis.results():
            for image_result in gis.results():
                filename = os.path.basename(image_result.path)
                downloaded_image_path = os.path.join(temp_dir, filename)
                downloaded_image_paths.append(downloaded_image_path)

        if downloaded_image_paths:
            selected_image_path = random.choice(downloaded_image_paths)

            image = imread(selected_image_path)
            return image
        else:
            logging.error("No se pudo descargar ninguna imagen.")
            return None
