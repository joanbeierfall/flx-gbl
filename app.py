import os
import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

# Initialize the image processor
base_path = "black-forest-labs/FLUX.1-dev"    
lora_base_path = "./models"

pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe.to("cuda")

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

def single_condition_generate_image(prompt, spatial_img_path, height, width, seed, control_type, output_image_path):
    # Load the spatial image
    spatial_img = Image.open(spatial_img_path)
    
    # Set the control type
    if control_type == "Ghibli":
        lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)
    
    # Process the image
    spatial_imgs = [spatial_img] if spatial_img else []
    image = pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed), 
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
    ).images[0]
    clear_cache(pipe.transformer)
    
    # Save the generated image to the output path
    image.save(output_image_path)
    print(f"Generated image saved to {output_image_path}")
    return image

# Example usage
prompt = "Ghibli Studio style, Charming hand-drawn anime-style illustration"
spatial_img_path = "./test_imgs/00.png"  # Path to the input image
height = 768
width = 768
seed = 42
control_type = "Ghibli"
output_image_path = "./output_image.png"  # Path where the generated image will be saved

# Call the function to generate the image and save it
single_condition_generate_image(prompt, spatial_img_path, height, width, seed, control_type, output_image_path)
