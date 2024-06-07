import gradio as gr
import os
import time

from models.Llama.llama import Llama
from src.aux_functions import apply_transformations, images_to_temp_paths, google_image_search
from src.model_manager import ModelManager
import logging


# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

def update_images(input_images): return input_images


def switch(options: list):
    """ Sky image input visibility"""
    if "Sky" in options:
        return gr.Image(visible=True)

    else:
        return gr.Image(visible=False)


def apply_transformations_event(input_images, options, sky_image_input):
    # Funcion para poder usar model_manager en el contexto Gradio, es una funcion auxiliar
    # Gradio ejecuta una funcion con todo argumentos de tipo Gradio y dentro de esta funcion ya usamos la funcion original y instancia de ModelManager

    enhanced_images = apply_transformations(input_images, options, model_manager, sky_image_input)
    return enhanced_images


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] != '':
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False, placeholder="Wait to LLM response...", show_label=False)


def add_response(history, response):
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history


def select_input_images(history):
    input_images = []
    for item in reversed(history):
        print("Item", item)
        if isinstance(item[0], tuple):
            input_images.append(item[0][0])
        # elif isinstance(item[0], str) and isinstance(item[1], str): # Si esta esto puesto solo coge imagenes hasta el ultimo [input_text, output_text]
        #     return input_images

    return input_images


def llm_bot(history):

    print("History", history)

    input_text = None
    for item in reversed(history):
        if isinstance(item[0], str):
            input_text = item[0]
            break

    input_images = select_input_images(history)

    if not input_images and input_text:
        print('Only text')
        response = llama_model.generate(input_text, False)
        for updated_history in add_response(history, response):
            yield updated_history

    elif input_images and not input_text:
        print('Only image')
        response = llama_model.generate(None, True)
        for updated_history in add_response(history, response):
            yield updated_history

    elif input_images and input_text:
        tasks = llama_model.generate(input_text, True)
        print("Response LLM", tasks)
        if tasks:
            if 'Sky' in tasks:
                response = "El cielo se va a reemplazar por " + str(tasks[-1])
                for updated_history in add_response(history, response):
                    yield updated_history
                sky = google_image_search(tasks)
                enhanced_images = apply_transformations(input_images, ['Sky'], model_manager, sky_image=sky)
            else:
                response = "La imagen se va a mejorar con " + str(tasks)
                for updated_history in add_response(history, response):
                    yield updated_history
                enhanced_images = apply_transformations(input_images, tasks, model_manager)

            output_img_paths = images_to_temp_paths(enhanced_images)

            for img_path in output_img_paths:
                history.append((None, (img_path,)))
                yield history
        else:
            response = 'Especifica mejor la tarea que quieres aplicar a la imagen.'
            for updated_history in add_response(history, response):
                yield updated_history

    else:

        response = llama_model.generate(None, False)
        history += [[None, None]]
        for updated_history in add_response(history, response):
            yield updated_history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_manager = ModelManager()
    llama_model = Llama()

    with gr.Blocks() as demo:
        with gr.Tab("Manual Editor"):
            input_images = gr.File(type="filepath", label="Input Images", file_count="multiple", file_types=["image"],
                                   interactive=True)

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
                ["Low Light", "Denoise", "Deblur", "White Balance", "Sky"], value=["White Balance"], multiselect=True,
                label="Choose the filters according to the order in which you want to apply them. "
            )

            # Sky image input (initially hidden)
            sky_image_input = gr.Image(
                "data/demo_images/skybox/galaxy.jpg",
                elem_id="sky_image_input",
                label="Sky Image",
                visible=False,
                height=500,  # Establece la altura máxima de la imagen
                width=10000  # Establece la anchura máxima de la imagen
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

            btn_submit.click(apply_transformations_event, [input_images, options, sky_image_input], output_gallery)

        with gr.Tab("ChatBot Editor"):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                height=750,

            )

            chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

            chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input]) # Event listener, starts when send button is used. Text box update to empty and chatbot is updated with user message and image.
            bot_msg = chat_msg.then(llm_bot, chatbot, chatbot, api_name="bot_response") # Then is executed when chat_msg finish (when add_message finish). llm_bot is called with the chatbot history and updated with the llm_respone
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False), None, [chat_input]) # Empty textbox after sending message

    demo.queue()
    demo.launch()
