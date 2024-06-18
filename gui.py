import gradio as gr
import logging

from models.Llama.llama import Llama
from src.aux_functions import *
from src.objects.model_manager import ModelManager

import configparser
config = configparser.ConfigParser()
config.read('data\config.ini')


def update_images(input_images): return input_images


# Definir una función para cargar todos los modelos
def load_all_models():
    model_manager.load_all_models()
    return gr.Button(value="Modelos cargados!", interactive=False, visible=True, variant="secondary"), gr.Textbox(visible=False)


def waiting_loading_models():
    return gr.Button("Cargando modelos...", interactive=False)


def show_progress_box():
    return gr.Textbox(value="Waiting", visible=True)


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
    history = process_message(history, message)
    return history, gr.MultimodalTextbox(value=None, interactive=False, placeholder="Wait to LLM response...", show_label=False)



def select_input_images_and_text(history):
    return extract_images_and_text(history)


def llm_bot(history):
    input_images, input_text = select_input_images_and_text(history)

    if input_images and input_text:
        yield from handle_image_and_text_input(history, input_images, input_text, llama_model, model_manager)
    elif input_images:
        yield from handle_image_input(history, llama_model)
    elif input_text:
        yield from handle_text_input(history, input_text, llama_model)
    else:
        yield from add_response(history, "Introduce una imagen y cómo deseas mejorarla, así te podré mostrar todas mis habilidades.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_manager = ModelManager()
    llama_model = Llama()

    with gr.Blocks() as demo:
        # Crear el botón con su estado inicial
        toggle_btn = gr.Button("Cargar modelos", variant="primary", interactive=True)
        progress_text_box = gr.Textbox(visible=False)

        # Asocia las funciones al evento de clic del botón
        toggle_btn.click(waiting_loading_models, outputs=toggle_btn, queue=False)
        toggle_btn.click(show_progress_box, outputs=progress_text_box, queue=False)
        toggle_btn.click(load_all_models, outputs=[toggle_btn, progress_text_box])

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
                value=config["gradio"]["background_image"],
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
