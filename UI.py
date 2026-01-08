import gradio as gr
import torch
import torchvision.transforms as tvt
from PIL import Image
import numpy as np
import cupy
from rembg import remove

from pipelines import DMVTONPipeline
from utils.torch_utils import get_ckpt, load_ckpt, select_device
from utils.general import warm_up

# Load the model once
device = select_device(0)  # Use GPU 0
pipeline = DMVTONPipeline(checkpoints={'warp': 'checkpoints/dmvton_pf_warp.pt', 'gen': 'checkpoints/dmvton_pf_gen.pt'})
pipeline.to(device)
pipeline.eval()

# Warm up
dummy_input = {
    'person': torch.randn(1, 3, 256, 192).to(device),
    'clothes': torch.randn(1, 3, 256, 192).to(device),
    'clothes_edge': torch.randn(1, 1, 256, 192).to(device),
}
with cupy.cuda.Device(int(str(device).split(':')[-1])):
    warm_up(pipeline, **dummy_input)

def preprocess_image(img, normalize=True):
    img = img.resize((192, 256))  # W x H
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else tvt.Lambda(lambda x: x)
    ])
    return transform(img).unsqueeze(0).to(device)

def preprocess_mask(cloth_img):
    # Use rembg to remove background and get alpha channel as mask
    cloth_rgba = remove(cloth_img)
    # Get alpha channel
    alpha = np.array(cloth_rgba)[:, :, 3]  # Shape: (H, W)
    mask_img = Image.fromarray(alpha, mode='L')
    mask_img = mask_img.resize((192, 256))
    transform = tvt.ToTensor()
    return transform(mask_img).unsqueeze(0).to(device)

def virtual_try_on(person_img, cloth_img):
    if person_img is None or cloth_img is None:
        return None
    
    person_tensor = preprocess_image(person_img, normalize=True)
    cloth_tensor = preprocess_image(cloth_img, normalize=True)
    mask_tensor = preprocess_mask(cloth_img)
    
    with torch.no_grad():
        with cupy.cuda.Device(int(str(device).split(':')[-1])):
            p_tryon, _ = pipeline(person_tensor, cloth_tensor, mask_tensor, phase="test")
    
    # Denormalize
    p_tryon = (p_tryon + 1) / 2
    p_tryon = torch.clamp(p_tryon, 0, 1)
    p_tryon = p_tryon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    p_tryon = p_tryon.astype(np.uint8)
    
    return Image.fromarray(p_tryon)

# Gradio interface
with gr.Blocks(title="DM-VTON Virtual Try-On") as demo:
    gr.Markdown("# DM-VTON Virtual Try-On Demo")
    gr.Markdown("Upload a person image and a garment image. The garment mask will be generated automatically.")
    
    with gr.Row():
        person_input = gr.Image(label="Person Image", type="pil")
        cloth_input = gr.Image(label="Garment Image", type="pil")
    
    submit_btn = gr.Button("Try On!")
    output_img = gr.Image(label="Try-On Result")
    
    submit_btn.click(
        virtual_try_on,
        inputs=[person_input, cloth_input],
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch()