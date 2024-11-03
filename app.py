import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageEnhance

# Load the Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"  # You can choose another model if you'd like
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

def generate_image(prompt, brightness):
    # Generate an image from the prompt
    with torch.no_grad():
        image = pipe(prompt).images[0]

    # Adjust the brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    return image

# Gradio Interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, label="Brightness")
    ],
    outputs="image",
    title="Image Generation with Diffusers",
    description="Enter a prompt to generate an image and adjust the brightness."
)

iface.launch()