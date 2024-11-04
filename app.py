import gradio as gr
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler, PNDMScheduler
import torch

# Load Stable Diffusion pipeline
model_id = "CompVis/stable-diffusion-v1-4"
default_scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=default_scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Scheduler options
schedulers = {
    "Photo-Realistic (PNDM) - Best for realistic details, moderate speed": PNDMScheduler,
    "Artistic & Imaginative (Euler Ancestral) - Best for creative scenes, moderate speed": EulerAncestralDiscreteScheduler,
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

    return image

# Warning function to show a message if the user selects a high value for quality
def show_warning(quality):
    if quality > 80:
        return "⚠️ High Quality: This setting may slow down generation and might not provide additional visual improvement. Consider using 50-80 steps for best results."
    else:
        return ""

# Build Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# ✨Wallpaper/Profile Picture Generator (Beta)")
        gr.Markdown("⚠️ **Live effects and advanced prompt engineering coming soon! Disclaimer**: Results may not always be accurate or perfectly aligned with your prompt. Experiment with prompt adjustments and settings to get the best results.")

    with gr.Row():
        # Controls on the left
        with gr.Column():
            # Custom prompt textbox
            custom_prompt = gr.Textbox(label="Enter Your Custom Prompt", lines=2, placeholder="Describe your wallpaper/profile picture (e.g., 'A forest at sunset')")

            # Category-based prompt controls
            genre = gr.Dropdown(["Futuristic", "Nature", "Abstract", "Fantasy", "Sci-Fi", "Cyberpunk"], label="Genre")
            style = gr.Dropdown(["Realistic", "Surreal", "Digital Art", "Cartoon", "Photorealistic"], label="Style")
            theme = gr.Dropdown(["Landscape", "Portrait", "Abstract Patterns", "Architecture"], label="Theme")
            lighting = gr.Dropdown(["Warm", "Cool", "Cinematic", "Soft", "Neon"], label="Lighting")

            # Quality (Detail) slider with warning message
            quality = gr.Slider(
                20, 150, value=80, step=10, 
                label="Image Quality (Detail)", 
                info="Higher values yield more detail but take longer to generate."
            )
            warning_message = gr.Markdown("")

            # Scheduler options with purpose-based descriptions
            scheduler_choice = gr.Dropdown(
                [
                "Photo-Realistic (PNDM) - Best for realistic details, moderate speed",
                "Artistic & Imaginative (Euler Ancestral) - Best for creative scenes, moderate speed",
                "High-Definition & Fast (DDIM) - Good quality with fastest speed"
                ],
                label="Artistic Style & Speed"
            )

            # Size option for wallpaper or profile picture
            size = gr.Dropdown(["Wallpaper", "Profile Picture"], label="Image Size", value="Wallpaper")

            # Generate button
            generate_button = gr.Button("Generate Image")

        # Image output on the right
        with gr.Column():
            output_image = gr.Image(label="Generated Image")

    # Display warning message for high-quality settings
    quality.change(show_warning, inputs=[quality], outputs=warning_message)

    # Bind the generate function to the generate button
    generate_button.click(
        fn=generate_image,
        inputs=[custom_prompt, genre, style, theme, lighting, scheduler_choice, quality, size],
        outputs=output_image
    )

# Launch with a public shareable link
demo.launch(share=True)