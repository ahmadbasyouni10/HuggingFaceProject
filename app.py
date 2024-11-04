import gradio as gr
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler, PNDMScheduler
import torch
from PIL import ImageEnhance, Image
import numpy as np

# Load Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
default_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=default_scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Scheduler options
schedulers = {
    "Artistic & Imaginative (Euler Ancestral) - Recommended for creative scenes, moderate speed": EulerAncestralDiscreteScheduler,
    "Photo-Realistic (PNDM) - Best for realistic details, moderate speed": PNDMScheduler,
    "High-Definition & Fast (DDIM) - Good quality with fastest speed": DDIMScheduler,
}

# Main image generation function with dynamic scheduling and size option
def generate_image(prompt, genre, style, theme, lighting, scheduler_choice, quality, size):
    # Combine custom prompt with selected categories
    prompt_text = (
        f"{prompt.strip()} in a {genre.lower()} wallpaper style, "
        f"with {style.lower()} visuals, focusing on a {theme.lower()} theme "
        f"and {lighting.lower()} lighting."
    )

    # Set the scheduler based on user choice
    scheduler = schedulers[scheduler_choice].from_pretrained(model_id, subfolder="scheduler")
    pipe.scheduler = scheduler

    # Set output size based on selection
    image_size = (512, 512) if size == "Profile Picture" else (1024, 768)

    # Generate image with specified quality and size
    with torch.no_grad():
        image = pipe(prompt_text, num_inference_steps=quality, guidance_scale=7.5).images[0]
        image = image.resize(image_size)  # Resize image to fit selected dimensions

    return np.array(image)  # Return as NumPy array for Gradio

# Post-processing function for brightness and contrast
def adjust_brightness_contrast(image, brightness, contrast):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return np.array(image)

# Warning function to show a message if the user selects a high value for quality
def show_warning(quality):
    if quality > 80:
        return "⚠️ High Quality: This setting may slow down generation and might not provide additional visual improvement. Consider using 50-80 steps for best results."
    return ""

# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# ✨ AI-Powered Wallpaper/Profile Picture Generator\n🖼️ A tool to generate and fine-tune AI-created wallpapers and profile pictures with adjustable styles and effects.")
    gr.Markdown("⚠️ **Live effects and advanced prompt engineering coming soon! Disclaimer**: Results may not always be accurate or perfectly aligned with your prompt. Experiment with prompt adjustments and settings to get the best results.")
    
    # Image Generation Section
    with gr.Tab("Image Generator"):
        gr.Markdown("## Generate an Image")
        
        with gr.Row():
            with gr.Column():
                custom_prompt = gr.Textbox(label="Custom Prompt", placeholder="Describe your image (e.g., 'A forest at sunset')")
                genre = gr.Dropdown(["Futuristic", "Nature", "Abstract", "Fantasy", "Sci-Fi", "Cyberpunk"], label="Genre")
                style = gr.Dropdown(["Realistic", "Surreal", "Digital Art", "Cartoon", "Photorealistic"], label="Style")
                theme = gr.Dropdown(["Landscape", "Portrait", "Abstract Patterns", "Architecture"], label="Theme")
                lighting = gr.Dropdown(["Warm", "Cool", "Cinematic", "Soft", "Neon"], label="Lighting")
                quality = gr.Slider(20, 150, value=80, step=10, label="Image Quality", info="Higher values yield more detail but take longer to generate.")
                warning_message = gr.Markdown("")
                
                # Set Euler Ancestral as the default option in the dropdown
                scheduler_choice = gr.Dropdown(
                    [
                    "Artistic & Imaginative (Euler Ancestral) - Recommended for creative scenes, moderate speed",
                    "Photo-Realistic (PNDM) - Best for realistic details, moderate speed",
                    "High-Definition & Fast (DDIM) - Good quality with fastest speed"
                    ],
                    label="Artistic Style & Speed",
                    value="Artistic & Imaginative (Euler Ancestral) - Recommended for creative scenes, moderate speed"  # Set as default
                )
                
                size = gr.Dropdown(["Wallpaper", "Profile Picture"], label="Image Size", value="Wallpaper")
                generate_button = gr.Button("Generate Image")

            with gr.Column():
                generated_image = gr.Image(label="Generated Image", interactive=False)

        # Display warning message for high-quality settings
        quality.change(show_warning, inputs=[quality], outputs=warning_message)

        # Bind the generate function to the generate button
        generate_button.click(
            fn=generate_image,
            inputs=[custom_prompt, genre, style, theme, lighting, scheduler_choice, quality, size],
            outputs=generated_image
        )

    # Post-Processing Section
    with gr.Tab("Edit Generated Image"):
        gr.Markdown("## Adjust Brightness & Contrast")
        
        with gr.Row():
            with gr.Column():
                brightness_slider = gr.Slider(0.5, 2.0, value=1.0, label="Brightness")
                contrast_slider = gr.Slider(0.5, 2.0, value=1.0, label="Contrast")
                apply_adjustments = gr.Button("Apply Adjustments")

            with gr.Column():
                output_image = gr.Image(label="Adjusted Image", interactive=False)

        # Bind the brightness and contrast adjustment function to the Apply Adjustments button
        apply_adjustments.click(
            fn=adjust_brightness_contrast,
            inputs=[generated_image, brightness_slider, contrast_slider],
            outputs=output_image
        )

# Launch with a public shareable link
demo.launch(share=True)