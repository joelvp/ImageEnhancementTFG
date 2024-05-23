import gradio as gr
import logging
from src.aux_functions import apply_transformations, image_to_base64, query_message

from src.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)

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


def llm_response(history,text,img):
    if not img:
        # response = txt_model.generate_content(text)
        # history += [(None,response.text)]
        response = "Respuesta a mensaje solo de texto + recordatorio de que esto es para mejorar imagenes"
        history += [(None,response)]
        return history

    else:
        # img = PIL.Image.open(img)
        # response = vis_model.generate_content([text,img])
        base64_img = image_to_base64(img) # Esto seria de la imagen que contesta el modelo
        data_url = f"data:image/jpeg;base64,{base64_img}"
        history += [(f"Imagen mejorada! ![]({data_url})", None)]
        return history

if __name__ == "__main__":
    
    model_manager = ModelManager()
    
    with gr.Blocks() as demo:
        with gr.Tab("Manual Editor"):
            gr.Markdown("IMAGE EDITOR")
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

            btn_submit.click(apply_transformations_event, [input_images, options, sky_image_input], output_gallery)
            
        with gr.Tab("ChatBot editor"):
            with gr.Row():
                image_box = gr.Image(type="filepath")

                chatbot = gr.Chatbot(
                    scale=2,
                    height=750
                )
            text_box = gr.Textbox(
                placeholder="Enter text and press enter, or upload an image",
                container=False,
            )

            btn = gr.Button("Submit")
            clicked = btn.click(query_message,
                                [chatbot, text_box, image_box],
                                chatbot
                                ).then(llm_response,
                                [chatbot,text_box,image_box],
                                chatbot
                                )
    demo.launch()
