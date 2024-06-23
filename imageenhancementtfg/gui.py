import gradio as gr
from typing import Dict
from src.utils import *
from src.objects.model_manager import ModelManager

HistoryItem: TypeAlias = Union[Tuple[str, str], Tuple[Optional[str], str], Tuple[str, Optional[str]], Tuple[Optional[Tuple[str]], Optional[str]]]
History: TypeAlias = List[HistoryItem]


def update_images(input_images: gr.File) -> gr.File:
    """
    Updates the input images displayed in the Gradio File component.

    Parameters:
            input_images (gr.File): The input file component from Gradio.

    Returns:
            gr.File: Updated input file component.
    """
    return input_images


# Definir una función para cargar todos los modelos
def load_all_models() -> tuple[gr.Button, gr.Textbox]:
    """
    Loads all models and returns a button and textbox.

    Returns:
            tuple[gr.Button, gr.Textbox]: Button indicating models are loaded and a hidden textbox.
    """
    model_manager.load_all_models()
    return gr.Button(value="Modelos cargados!", interactive=False, visible=True, variant="secondary"), gr.Textbox(visible=False)


def waiting_loading_models() -> gr.Button:
    """
    Returns a button indicating models are being loaded.

    Returns:
            gr.Button: Button indicating models are being loaded.
    """
    return gr.Button("Cargando modelos...", interactive=False)


def show_progress_box() -> gr.Textbox:
    """
    Returns a textbox displaying progress.

    Returns:
            gr.Textbox: Textbox displaying progress.
    """
    return gr.Textbox(value="Waiting", visible=True)


def switch(options: list[str]) -> gr.Image:
    """
    Controls visibility of Sky image based on options.

    Parameters:
            options (list[str]): List of options selected.

    Returns:
            gr.Image: Image component either visible or hidden based on 'Sky' option.
    """
    if "Sky" in options:
        return gr.Image(visible=True)

    else:
        return gr.Image(visible=False)


def apply_transformations_event(input_images: List[str], options: List[str], sky_image_input: Optional[np.ndarray]) -> List[np.ndarray]:
    """
        Event handler to apply transformations based on user inputs.

        Parameters:
                input_images (List[str]: Input images from user.
                options (List[str]): List of transformation options.
                sky_image_input (Optional[np.ndarray]): Sky image input component.

        Returns:
                List[np.ndarray]: List of enhanced images as NumPy arrays.
        """
    enhanced_images = apply_transformations(input_images, options, model_manager, sky_image_input)
    return enhanced_images


def add_message(history: History, message: Dict) -> tuple[History, gr.MultimodalTextbox]:
    """
    Adds the user message to the chat history and updates the input textbox.

    Parameters:
            history (History): Chat history.
            message (Dict): "text" and "files", both optional, of the user input
    Returns:
            tuple[History, gr.MultimodalTextbox]: Updated history and a MultimodalTextbox component.
    """
    history = process_message(history, message)
    return history, gr.MultimodalTextbox(value=None, interactive=False, placeholder="Wait to LLM response...", show_label=False)


def select_input_images_and_text(history: History) -> Tuple[list, str]:
    """
    Selects input images and text from chat history.

    Parameters:
            history (History): Chat history.

    Returns:
            Tuple[list, str]: List of input images and input text.
    """
    return extract_images_and_text(history)


def llm_bot(history: History) -> Iterator[History]:
    """
        Generates responses with LlamaAPI based on the user input.

        Parameters:
                history (History): Chat history.

        Returns:
                Iterator[History]: Responses with LlamaAPI.

        """
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

        toggle_btn = gr.Button("Cargar modelos", variant="primary", interactive=True)
        progress_text_box = gr.Textbox(visible=False)

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
                columns=3,
                rows=1,
                object_fit="contain",
                height="auto"
            )

            input_images.change(update_images, [input_images], input_gallery)

            options = gr.Dropdown(
                ["Low Light", "Denoise", "Deblur", "White Balance", "Sky", "Fish Eye"], value=["White Balance"], multiselect=True,
                label="Choose the filters according to the order in which you want to apply them. "
            )

            # Sky image input (initially hidden)
            sky_image_input = gr.Image(
                value=config["gradio"]["background_image"],
                elem_id="sky_image_input",
                label="Sky Image",
                visible=False,
                height=500,
                width=10000
            )

            # Sky image input depends on Sky CheckBox
            options.change(switch, options, sky_image_input)

            btn_submit = gr.Button("Enhance", variant="primary")

            output_gallery = gr.Gallery(
                label="Output Images",
                elem_id="output_gallery",
                format="png",
                show_label=False,
                columns=3,
                rows=1,
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
