import gradio as gr
from PIL import Image
from models.Deep_White_Balance.PyTorch.white_balance import white_balance_gui
# from models.LLFlow.code.lowlight import lowlight

def image_enhancement(input_images, options):
    enhanced_images = []

    for input_image_path in input_images:
        input_image = Image.open(input_image_path)

        result_image = input_image
        if "White Balance" in options:
            print("Applying White Balance")
            white_balance_gui(input_images)

        if "Low Light" in options:
            print("Applying Low Light")
            # lowlight()

        if "Blur" in options:
            print("Applying Blur")

        enhanced_images.append(result_image)

    return enhanced_images

def show_original_images(original_images):

    return original_images

with gr.Blocks() as demo:
    input_images = gr.File(type="filepath", label="Input Images", file_count="multiple", file_types=["image"], interactive=True)
    
    btn_show = gr.Button("Show Original Images", scale=0)
    
    input_gallery = gr.Gallery(
        label="Input Images",
        elem_id="input_gallery",
        show_label=False,
        columns=[3],
        rows=[1],
        object_fit="contain",
        height="auto"
    )
    
    btn_show.click(show_original_images, [input_images] , input_gallery)

    options = gr.CheckboxGroup(["White Balance", "Low Light", "Blur"], label="Options", info="Enhance the image")
    
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

    btn_submit.click(image_enhancement, [input_images, options], output_gallery)

if __name__ == "__main__":
    demo.launch()
