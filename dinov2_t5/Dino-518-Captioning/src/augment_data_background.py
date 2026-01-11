"""
DeepFashion 数据增强脚本 [实景背景合成版]
硬件环境: NVIDIA RTX 5090 (32GB VRAM)
核心功能: 
1. BiRefNet 分割: 提取服装模特前景（扣图）
2. LCM-LoRA 加速: 仅需 8 步即可生成高质量实景背景
3. 实景提示词库: 模拟街拍、家居等多种真实场景，提升模型泛化能力
4. 像素级合成: 前景与生成后的背景进行平滑融合
"""

import argparse
import random
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from diffusers import LCMScheduler, StableDiffusionInpaintPipeline
from transformers import AutoModelForImageSegmentation

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "aug_images"

# 模型路径 (建议指向本地，或者使用 HuggingFace ID)
SEG_MODEL_PATH = PROJECT_ROOT / "models" / "birefnet" 
SD_MODEL_PATH = PROJECT_ROOT / "models" / "sd-inpainting"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

# 核心参数 (针对 5090 优化)
BATCH_SIZE = 48       
NUM_WORKERS = 16       

# 分辨率设置
SD_RESOLUTION = 512
RMBG_INPUT_SIZE = 512 
DINOV2_RESOLUTION = 518 

# LCM 加速设置
NUM_INFERENCE_STEPS = 8 
GUIDANCE_SCALE = 2.0 

# 背景混合强度 (1.0 = 完全替换背景，不保留原图白底)
# 真实场景必须设为 1.0，否则画面会发白失真
BACKGROUND_OPACITY = 1.0 

SEED = 42

# --- 真实场景提示词库 (用于生成背景) ---
REALISTIC_SCENES = [
    # 1. 城市/街道 (Urban/Street) - 模拟街拍
    "a busy city street sidewalk in New York, blurred cars and yellow taxis in background, daytime, natural sunlight, urban atmosphere",
    "a quiet european cobblestone street with old brick buildings, soft afternoon light, depth of field, street photography",
    "a modern city plaza with glass skyscrapers reflecting blue sky, sunny day, blurred background, bokeh",
    "an urban alleyway with textured brick walls and soft shadows, cinematic lighting, realistic texture",
    "a pedestrian crossing in Tokyo with blurred city lights, overcast soft lighting, urban vibe",
    "a fashion district street with luxury store fronts in background, blurred, high fashion photography",
    "a sunny sidewalk with tree shadows cast on the ground, summer vibes, blurred street",
    "a london street corner with red bus blurred in background, cloudy day, soft lighting",
    "a concrete staircase in an urban environment, industrial style, natural light",
    
    # 2. 室内/家居 (Indoor/Home) - 模拟居家/私服
    "a cozy modern living room with a beige sofa and sunlight streaming through the sheer curtains, blurred interior",
    "a luxury walk-in closet with wooden shelves and warm lighting, depth of field, interior design",
    "a bright modern kitchen island background, clean and tidy, soft window light, blurred",
    "a minimalist bedroom with a white bed and floor lamp, morning light, cozy atmosphere, home photography",
    "an elegant hotel lobby with marble floors and warm chandelier lights, blurred background",
    "a sun-drenched hallway with wooden floors and white walls, clean architecture, depth of field",
    "a modern loft apartment with exposed brick walls and large windows, soft daylight",
    "a vintage room with patterned wallpaper and wooden furniture, warm retro lighting, blurred",
    
    # 3. 休闲/商业 (Leisure/Commercial) - 模拟探店/商场
    "a modern coffee shop interior with wooden tables and warm hanging lights, bokeh background, cafe vibe",
    "a luxury boutique store interior with clothing racks and mirrors, bright lighting, shopping mall context",
    "an art gallery hallway with white walls and soft track lighting, clean depth of field",
    "a modern office space with glass walls and desks, blurred background, professional atmosphere",
    "a library aisle with wooden bookshelves, soft quiet lighting, academic atmosphere, blurred",
    "a trendy restaurant interior with blurred tables and warm ambient lighting",
    "a shopping mall atrium with glass roof and natural light, blurred background",
    
    # 4. 自然/户外 (Nature/Outdoor) - 模拟外景
    "a sunny public park with green grass and oak trees, soft sunlight filtering through leaves, bokeh, nature photography",
    "a beautiful beach with white sand and blue ocean in the distance, sunny day, vacation vibe, blurred horizon",
    "a garden path with colorful flowers and greenery, soft spring light, depth of field",
    "a wooden deck patio with potted plants, outdoor sunlight, natural shadows",
    "a forest trail with sunlight filtering through canopy, nature background, soft focus",
    "a snowy street with soft white snow, winter fashion vibe, cold lighting, blurred",
    "a flower field with blurred colorful blooms, spring atmosphere, soft lighting",
    "a seaside promenade with railings and sea view, windy sunny day, blurred background",
    
    # 5. 极简/纹理 (Texture/Minimalist) - 模拟型录
    "a professional photo studio background with textured grey canvas backdrop, soft studio lighting",
    "a clean white brick wall with natural side lighting, minimalist fashion photography",
    "a raw concrete wall with industrial texture, soft shadows, modern aesthetic",
    "a warm beige plaster wall with soft window light shadows, mediterranean style",
    "an abstract blurred gradient background with warm tones, studio photography style"
]

# 通用画质增强后缀
QUALITY_SUFFIX = ", highly detailed, photorealistic, 8k, raw photo, depth of field, bokeh, sharp focus on foreground subject"

# 负面提示词
NEGATIVE_PROMPT = (
    "cartoon, anime, painting, drawing, illustration, 3d render, low quality, blurry foreground, "
    "distorted, ugly, messy, text, watermark, bad anatomy, extra limbs, missing limbs, people in background, cars in foreground"
)

# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def generate_random_prompt() -> str:
    """随机选择一个具体场景，并加上画质增强后缀"""
    scene = random.choice(REALISTIC_SCENES)
    return f"{scene}{QUALITY_SUFFIX}"

def tensor_mask_to_pils(mask_tensor: torch.Tensor) -> List[Image.Image]:
    mask_tensor = mask_tensor.detach().clamp(0.0, 1.0)
    mask_tensor = mask_tensor.squeeze(1).cpu().numpy()
    mask_tensor = (mask_tensor * 255.0).astype(np.uint8)
    return [Image.fromarray(mask, mode="L") for mask in mask_tensor]

def make_center_mask(size: int = SD_RESOLUTION, ratio: float = 0.6) -> torch.Tensor:
    mask = torch.zeros(1, size, size, dtype=torch.float32)
    margin = int(size * (1 - ratio) / 2)
    mask[:, margin:size - margin, margin:size - margin] = 1.0
    return mask

# ============================================================
# Dataset
# ============================================================

class DeepFashionDataset(Dataset):
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, image_dir: Path) -> None:
        self.image_paths = sorted(
            [p for ext in self.IMG_EXTENSIONS for p in image_dir.glob(f"*{ext}")]
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        self.resize_512 = transforms.Resize(
            (SD_RESOLUTION, SD_RESOLUTION), interpolation=transforms.InterpolationMode.LANCZOS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Tuple[Image.Image, str]]:
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_512 = self.resize_512(img)
            return img_512, path.name
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    images, filenames = zip(*batch)
    return list(images), list(filenames)

# ============================================================
# Segmentation Class (FIXED)
# ============================================================

class GPUSegmenter:
    def __init__(self, device: torch.device) -> None:
        print(f"[INFO] Loading segmentation model: {SEG_MODEL_PATH}")
        self.device = device
        # 优先加载本地模型，若无则尝试下载
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                SEG_MODEL_PATH,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)
        except:
            print("[WARN] Local model not found, downloading from HuggingFace...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)
        self.model.eval()

    @staticmethod
    def _preprocess(images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in images:
            resized = img.resize((RMBG_INPUT_SIZE, RMBG_INPUT_SIZE), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # Normalize
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
            tensors.append(torch.from_numpy(arr))
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _extract_tensor_from_output(output):
        if isinstance(output, torch.Tensor): return output
        if isinstance(output, (list, tuple)): return GPUSegmenter._extract_tensor_from_output(output[0])
        if hasattr(output, "logits"): return GPUSegmenter._extract_tensor_from_output(output.logits)
        if hasattr(output, "out"): return GPUSegmenter._extract_tensor_from_output(output.out)
        raise TypeError(f"Unexpected output type: {type(output)}")

    @torch.no_grad()
    def process_batch(self, images_512: List[Image.Image]) -> torch.Tensor:
        pixel_values = self._preprocess(images_512).to(self.device, dtype=torch.float16)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            raw_output = self.model(pixel_values)
        
        logits = self._extract_tensor_from_output(raw_output).float()
        mask_fg = torch.sigmoid(logits)
        
        # Resize to 512 for SD
        mask_fg = F.interpolate(
            mask_fg,
            size=(SD_RESOLUTION, SD_RESOLUTION),
            mode="bilinear",
            align_corners=False
        )
        mask_fg = (mask_fg > 0.5).float()
        return mask_fg.clamp(0.0, 1.0)

# ============================================================
# Compositing Logic
# ============================================================

def composite_images(
    original_pil: Image.Image,
    generated_pil: Image.Image,
    mask_fg_tensor: torch.Tensor,
    bg_opacity: float = 1.0
) -> Image.Image:
    if mask_fg_tensor.dim() == 2: mask_nchw = mask_fg_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_fg_tensor.dim() == 3: mask_nchw = mask_fg_tensor.unsqueeze(0)
    else: raise ValueError(f"Unexpected mask dims: {mask_fg_tensor.shape}")

    # Resize everything to 518 (DINOv2 input size)
    mask_fg_518 = F.interpolate(
        mask_nchw,
        size=(DINOV2_RESOLUTION, DINOV2_RESOLUTION),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()
    mask_fg_518 = np.clip(mask_fg_518, 0.0, 1.0)

    original_518 = original_pil.resize((DINOV2_RESOLUTION, DINOV2_RESOLUTION), Image.Resampling.LANCZOS)
    generated_518 = generated_pil.resize((DINOV2_RESOLUTION, DINOV2_RESOLUTION), Image.Resampling.LANCZOS)

    orig_np = np.array(original_518, dtype=np.float32) / 255.0
    gen_np = np.array(generated_518, dtype=np.float32) / 255.0
    mask_np = mask_fg_518[..., None]

    # Background Blending (With Opacity)
    # If opacity=1.0, blended_bg = gen_np (Full replacement)
    blended_bg = gen_np * bg_opacity + orig_np * (1.0 - bg_opacity)
    
    # Final Composite: Foreground from Original + Background from Generated
    comp = orig_np * mask_np + blended_bg * (1.0 - mask_np)
    
    comp = (comp * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(comp)

# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepFashion Turbo Augmentation")
    parser.add_argument("--input_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not detected.")

    device = torch.device("cuda")
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    dataset = DeepFashionDataset(args.input_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 2. Models
    segmenter = GPUSegmenter(device)

    print(f"[INFO] Loading Stable Diffusion + LCM-LoRA from: {SD_MODEL_PATH}")
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL_PATH,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
    except Exception as e:
        print(f"Error loading local SD model: {e}")
        print("Fallback to downloading from HuggingFace...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
    
    # LCM Injection
    try:
        pipe.load_lora_weights(LCM_LORA_ID)
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print("[INFO] LCM-LoRA loaded successfully.")
    except Exception as e:
        print(f"[WARN] Failed to load LCM-LoRA ({e}). Running in standard mode.")
        global NUM_INFERENCE_STEPS, GUIDANCE_SCALE
        NUM_INFERENCE_STEPS = 25
        GUIDANCE_SCALE = 7.5

    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)

    total = len(dataset)
    print("=" * 72)
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Processing {total} images...")
    print(f"[INFO] BG Opacity: {BACKGROUND_OPACITY} (Full Replacement)")
    print("=" * 72)

    progress = tqdm(dataloader, desc="Augmenting", unit="batch")
    
    for batch in progress:
        if batch is None: continue

        images_512, filenames = batch
        prompts = [generate_random_prompt() for _ in images_512]

        try:
            # A) Segmentation
            mask_fg = segmenter.process_batch(images_512)
            mask_fg_cpu = mask_fg.detach().cpu()
            
            mask_bg_for_sd = (1.0 - mask_fg_cpu).clamp(0.0, 1.0)
            
            # Empty mask check
            for i in range(mask_bg_for_sd.shape[0]):
                if mask_bg_for_sd[i].sum() < 100:
                    mask_bg_for_sd[i] = make_center_mask()

            mask_pils = tensor_mask_to_pils(mask_bg_for_sd)

            # B) SD Inpainting
            generators = [torch.Generator(device=device).manual_seed(random.randrange(1, 2**31 - 1)) for _ in range(len(images_512))]
            
            outputs = pipe(
                prompt=prompts,
                image=images_512,
                mask_image=mask_pils,
                negative_prompt=[NEGATIVE_PROMPT] * len(images_512),
                num_inference_steps=NUM_INFERENCE_STEPS, 
                guidance_scale=GUIDANCE_SCALE,           
                height=SD_RESOLUTION,
                width=SD_RESOLUTION,
                generator=generators
            ).images

            # C) Compositing
            for idx, (orig_pil, gen_pil, fname) in enumerate(zip(images_512, outputs, filenames)):
                final_img = composite_images(
                    orig_pil, 
                    gen_pil, 
                    mask_fg_cpu[idx], 
                    bg_opacity=BACKGROUND_OPACITY
                )
                final_img.save(args.output_dir / fname, format="JPEG", quality=95, subsampling=1)

        except Exception as exc:
            print(f"[ERROR] Batch failed: {exc}")
            import traceback
            traceback.print_exc()

    print("[INFO] Augmentation completed successfully.")

if __name__ == "__main__":
    main()