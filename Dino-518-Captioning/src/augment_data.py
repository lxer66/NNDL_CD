"""
High-Performance Data Augmentation Script for DeepFashion Dataset
Uses Stable Diffusion Inpainting to replace white backgrounds with unique, simple backgrounds
Optimized for RTX 5090 (32GB VRAM)
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline
from rembg import remove
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "images_augmented"

# Model
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"

# Processing
BATCH_SIZE = 32  # Can be increased if VRAM allows
NUM_WORKERS = 8
SD_RESOLUTION = 512  # Stable Diffusion requirement
DINOV2_RESOLUTION = 518  # Final output resolution
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30

# Quality
SEED = 42

# Vocabulary Pools for Dynamic Prompt Generation
LIGHTING = [
    "soft lighting", "cinematic lighting", "natural sunlight", 
    "studio light", "shadowy", "bright", "morning light"
]

COLORS = [
    "beige", "grey", "white", "cream", "light blue", 
    "pastel", "neutral", "warm tone", "cool tone"
]

TEXTURES = [
    "concrete wall", "brick wall", "wooden texture", 
    "fabric texture", "marble wall", "plaster wall", "painted wall"
]

SCENES = [
    "background", "studio background", "room corner", 
    "blurred outdoor background", "abstract background"
]

STYLES = [
    "minimalist", "clean", "modern", "bokeh", "out of focus"
]

NEGATIVE_PROMPT = (
    "people, crowd, animals, text, watermark, bad quality, "
    "messy, complex, ugly, deformed, distorted, blurry person"
)


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_random_prompt() -> str:
    """
    Generate a unique background prompt using combinatorial approach
    Returns a structured prompt: "{Lighting} {Color} {Texture} {Scene}, {Style}"
    """
    lighting = random.choice(LIGHTING)
    color = random.choice(COLORS)
    texture = random.choice(TEXTURES)
    scene = random.choice(SCENES)
    style = random.choice(STYLES)
    
    prompt = f"{lighting}, {color} {texture} {scene}, {style}, high quality, professional"
    return prompt


def resize_image(image: Image.Image, size: int) -> Image.Image:
    """Resize image to square size while maintaining aspect ratio"""
    # Use LANCZOS for high-quality downsampling
    return image.resize((size, size), Image.Resampling.LANCZOS)


def create_mask(image: Image.Image) -> Image.Image:
    """
    Remove background using rembg and create binary mask
    Returns: PIL Image where White=Background (to inpaint), Black=Foreground (to keep)
    """
    # Remove background (returns RGBA image with transparent background)
    output = remove(image)
    
    # Extract alpha channel as mask
    if output.mode == 'RGBA':
        alpha = output.split()[3]
    else:
        # Fallback if not RGBA
        alpha = Image.new('L', output.size, 255)
    
    # Invert mask: rembg gives 255 for foreground, 0 for background
    # We need: 255 for background (to inpaint), 0 for foreground (to keep)
    mask = Image.eval(alpha, lambda x: 255 - x)
    
    # Convert to RGB (SD requires 3-channel mask)
    mask = mask.convert('RGB')
    
    return mask


# ============================================================
# Dataset & DataLoader
# ============================================================

class AugmentationDataset(Dataset):
    """Dataset for loading and preprocessing images"""
    
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        self.image_paths = sorted(list(image_dir.glob("*.jpg"))) + \
                          sorted(list(image_dir.glob("*.png")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Optional[Tuple]:
        """
        Returns: (original_image_512, mask_512, image_path, prompt) or None if failed
        """
        image_path = self.image_paths[idx]
        
        try:
            # Read original image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to SD resolution (512x512)
            image_512 = resize_image(image, SD_RESOLUTION)
            
            # Generate mask using rembg (CPU-heavy operation)
            mask_512 = create_mask(image_512)
            
            # Generate unique prompt for this image
            prompt = generate_random_prompt()
            
            return image_512, mask_512, image_path, prompt
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    

def collate_fn(batch: List[Optional[Tuple]]) -> Tuple:
    """
    Custom collate function to filter out None values and batch data
    Returns: (images, masks, paths, prompts)
    """
    # Filter out failed samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return [], [], [], []
    
    images, masks, paths, prompts = zip(*batch)
    return list(images), list(masks), list(paths), list(prompts)


# ============================================================
# Main Augmentation Pipeline
# ============================================================

class BackgroundAugmentor:
    """Main class for background augmentation using Stable Diffusion"""
    
    def __init__(
        self,
        model_id: str = SD_MODEL_ID,
        device: str = "cuda",
        use_fp16: bool = True
    ):
        self.device = device
        self.use_fp16 = use_fp16
        
        print(f"Loading Stable Diffusion Inpainting model: {model_id}")
        print(f"Device: {device}, FP16: {use_fp16}")
        
        # Load pipeline
        torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disable for speed
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(device)
        
        # Enable memory optimizations
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ XFormers memory efficient attention enabled")
            except Exception:
                print("⚠ XFormers not available, using default attention")
        
        print("✓ Model loaded successfully\n")
    
    @torch.no_grad()
    def inpaint_batch(
        self,
        images: List[Image.Image],
        masks: List[Image.Image],
        prompts: List[str]
    ) -> List[Image.Image]:
        """
        Inpaint a batch of images
        Returns: List of inpainted images (512x512)
        """
        # Run batch inference
        outputs = self.pipe(
            prompt=prompts,
            image=images,
            mask_image=masks,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            negative_prompt=[NEGATIVE_PROMPT] * len(prompts)
        ).images
        
        return outputs
    
    def process_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS
    ):
        """Process entire dataset with batching"""
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset and dataloader
        dataset = AugmentationDataset(input_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        print("=" * 80)
        print("Starting Background Augmentation")
        print("=" * 80)
        print(f"Input Directory: {input_dir}")
        print(f"Output Directory: {output_dir}")
        print(f"Total Images: {len(dataset)}")
        print(f"Batch Size: {batch_size}")
        print(f"Num Workers: {num_workers}")
        print("=" * 80 + "\n")
        
        # Process batches
        total_processed = 0
        total_failed = 0
        
        progress_bar = tqdm(dataloader, desc="Augmenting")
        
        for batch_images, batch_masks, batch_paths, batch_prompts in progress_bar:
            
            if len(batch_images) == 0:
                continue
            
            try:
                # Inpaint batch (512x512)
                inpainted_images = self.inpaint_batch(
                    batch_images,
                    batch_masks,
                    batch_prompts
                )
                
                # Save each image
                for img_512, path in zip(inpainted_images, batch_paths):
                    try:
                        # Resize to DINOv2 resolution (518x518)
                        img_518 = resize_image(img_512, DINOV2_RESOLUTION)
                        
                        # Save with same filename
                        output_path = output_dir / path.name
                        img_518.save(output_path, quality=95)
                        
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"\nFailed to save {path.name}: {e}")
                        total_failed += 1
                
                progress_bar.set_postfix({
                    'Processed': total_processed,
                    'Failed': total_failed
                })
                
            except Exception as e:
                print(f"\nBatch processing error: {e}")
                total_failed += len(batch_images)
                continue
        
        print("\n" + "=" * 80)
        print("✅ Augmentation Complete!")
        print("=" * 80)
        print(f"Successfully processed: {total_processed}")
        print(f"Failed: {total_failed}")
        print(f"Output saved to: {output_dir}")
        print("=" * 80)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Background Augmentation for DeepFashion Dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DATA_DIR),
        help="Input directory containing original images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for augmented images"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available, using CPU (will be slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    # Create augmentor
    augmentor = BackgroundAugmentor(
        model_id=SD_MODEL_ID,
        device=device,
        use_fp16=(device == "cuda")
    )
    
    # Process dataset
    augmentor.process_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()