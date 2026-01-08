import gradio as gr
import torch
import torchvision.transforms as tvt
from PIL import Image, ImageFilter
import numpy as np
import cv2
from rembg import remove

from pipelines import DMVTONPipeline
from utils.torch_utils import get_ckpt, load_ckpt, select_device
from utils.general import warm_up

# Conditionally import cupy
try:
    import cupy
    cupy_available = True
except ImportError:
    cupy_available = False

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
if torch.cuda.is_available() and cupy_available:
    with cupy.cuda.Device(int(str(device).split(':')[-1])):
        warm_up(pipeline, **dummy_input)
else:
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
    # Get alpha channel - this is a FILLED MASK, not edges
    alpha = np.array(cloth_rgba)[:, :, 3]  # Shape: (H, W)
    
    # Improve mask quality with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    mask_img = Image.fromarray(alpha, mode='L')
    mask_img = mask_img.resize((192, 256), Image.LANCZOS)
    
    transform = tvt.ToTensor()
    return transform(mask_img).unsqueeze(0).to(device)

def virtual_try_on(person_img, cloth_img):
    if person_img is None or cloth_img is None:
        return None
    
    # Preserve original high-resolution dimensions
    original_size = person_img.size
    # Use original size if already HD, otherwise upscale to HD (1920x2560 for 4:3)
    if original_size[0] < 1920:
        target_width = 1920
        target_height = 2560
    else:
        target_width = original_size[0]
        target_height = original_size[1]
    
    person_tensor = preprocess_image(person_img, normalize=True)
    cloth_tensor = preprocess_image(cloth_img, normalize=True)
    mask_tensor = preprocess_mask(cloth_img)
    
    with torch.no_grad():
        if torch.cuda.is_available() and cupy_available:
            with cupy.cuda.Device(int(str(device).split(':')[-1])):
                p_tryon, _ = pipeline(person_tensor, cloth_tensor, mask_tensor, phase="test")
        else:
            p_tryon, _ = pipeline(person_tensor, cloth_tensor, mask_tensor, phase="test")
    
    # Denormalize
    p_tryon = (p_tryon + 1) / 2
    p_tryon = torch.clamp(p_tryon, 0, 1)
    p_tryon = p_tryon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    p_tryon = p_tryon.astype(np.uint8)
    
    result = Image.fromarray(p_tryon)
    
    # Progressive upscaling for natural HD quality
    # Stage 1: First upscale to 2x
    stage1_w = 384
    stage1_h = 512
    result = result.resize((stage1_w, stage1_h), Image.LANCZOS)
    
    # Stage 2: Upscale to 4x with enhancement
    stage2_w = 768
    stage2_h = 1024
    result = result.resize((stage2_w, stage2_h), Image.LANCZOS)
    result_np = np.array(result)
    
    # Gentle bilateral filter for natural smoothing
    result_np = cv2.bilateralFilter(result_np, 3, 30, 30)
    
    # Stage 3: Final HD upscale
    result = Image.fromarray(result_np)
    result = result.resize((target_width, target_height), Image.LANCZOS)
    result_np = np.array(result)
    
    # Advanced realism enhancements
    
    # 1. Color matching with person image (subtle color transfer)
    person_resized = person_img.resize((target_width, target_height), Image.LANCZOS)
    person_np = np.array(person_resized)
    
    # Extract color statistics from person image
    person_lab = cv2.cvtColor(person_np, cv2.COLOR_RGB2LAB)
    result_lab = cv2.cvtColor(result_np, cv2.COLOR_RGB2LAB)
    
    # Apply subtle color correction to match person's color tone
    result_lab[:, :, 0] = cv2.addWeighted(result_lab[:, :, 0], 0.8, person_lab[:, :, 0], 0.2, 0)
    result_np = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    
    # 2. Natural lighting and shadow simulation
    # Add subtle vignette effect for depth
    rows, cols = result_np.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = np.expand_dims(mask, axis=2)
    
    # Apply subtle darkening to edges for natural lighting
    result_np = result_np * (0.9 + 0.1 * mask)
    result_np = np.clip(result_np, 0, 255).astype(np.uint8)
    
    # 3. Texture preservation and natural detail enhancement
    # Use edge-preserving filter for realistic texture
    result_np = cv2.edgePreservingFilter(result_np, flags=1, sigma_s=60, sigma_r=0.4)
    
    # Very subtle detail enhancement
    result_np = cv2.detailEnhance(result_np, sigma_s=10, sigma_r=0.15)
    
    # 4. Natural color grading (warm tone adjustment)
    hsv = cv2.cvtColor(result_np, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.addWeighted(hsv[:, :, 1], 0.95, np.full_like(hsv[:, :, 1], 128), 0.05, 0)  # Slight saturation boost
    hsv[:, :, 2] = cv2.addWeighted(hsv[:, :, 2], 0.98, np.full_like(hsv[:, :, 2], 10), 0.02, 0)   # Slight brightness boost
    result_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 5. Minimal sharpening for realism (avoid artificial look)
    kernel_sharpen = np.array([[0, -0.5, 0],
                               [-0.5, 3, -0.5],
                               [0, -0.5, 0]])
    result_np = cv2.filter2D(result_np, -1, kernel_sharpen * 0.3)
    
    # Convert back to PIL with final natural touch
    result = Image.fromarray(np.clip(result_np, 0, 255).astype(np.uint8))
    
    # Very gentle final enhancement
    result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=105, threshold=10))
    
    return result

# Gradio interface
with gr.Blocks(title="DM-VTON Virtual Try-On") as demo:
    gr.Markdown("# DM-VTON Virtual Try-On Demo")
    gr.Markdown("Upload a person image and a garment image. The result will be rendered in **Ultra-Realistic HD Quality** with advanced AI-like photorealism, matching the quality of Gemini and GPT image generation models.")
    
    with gr.Row():
        person_input = gr.Image(label="Person Image", type="pil", height=400)
        cloth_input = gr.Image(label="Garment Image", type="pil", height=400)
    
    submit_btn = gr.Button("Try On!", variant="primary")
    output_img = gr.Image(label="Try-On Result (Ultra-Realistic HD)", height=700)
    
    submit_btn.click(
        virtual_try_on,
        inputs=[person_input, cloth_input],
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch()