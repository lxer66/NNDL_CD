import os
import sys

# ================= 1. 解决网络代理冲突 (必须放在最开头) =================
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

import torch
import json
import gradio as gr
from pathlib import Path
from PIL import Image

# ================= 2. 核心路径配置 =================
BASE_DIR = Path(__file__).parent

# 1. resnet_lstm_attention 路径
DIR_RESNET = str(BASE_DIR / "resnet_lstm_attention")

# 2. dinov2_t5 源代码路径
DIR_DINO_ROOT = BASE_DIR / "dinov2_t5"
DIR_DINO_SRC = str(DIR_DINO_ROOT / "Dino-518-Captioning" / "src")

CONFIG = {
    "Resnet_Model": {
        "model_path": os.path.join(DIR_RESNET, "13_checkpoint_deepfashion_aug_1_cap_per_img_5_min_word_freq.pth.tar"),
        "word_map_path": os.path.join(DIR_RESNET, "WORDMAP_deepfashion_aug_1_cap_per_img_5_min_word_freq.json")
    },
    "DINO_Model": {
        "vision_model": str(DIR_DINO_ROOT / "models" / "dinov2-base"),
        "text_model": str(DIR_DINO_ROOT / "models" / "flan-t5-base"),
        "lora_weights": str(DIR_DINO_ROOT / "weights" / "best_model")
    }
}

# 全局变量缓存模型
global_models = {
    "resnet": {"encoder": None, "decoder": None, "word_map": None, "rev_word_map": None},
    "dino": {"captioner": None}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 3. 模型加载逻辑 =================

def load_resnet_model():
    """加载 resnet_lstm_attention 模型"""
    if global_models["resnet"]["encoder"] is not None:
        return
    
    print("正在加载 resnet_lstm_attention 模型...")
    # 添加路径以便能导入 caption.py
    if DIR_RESNET not in sys.path:
        sys.path.append(DIR_RESNET)
    
    model_path = CONFIG["Resnet_Model"]["model_path"]
    word_map_path = CONFIG["Resnet_Model"]["word_map_path"]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    # weights_only=False 解决旧版 PyTorch 权重加载报错问题
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    global_models["resnet"]["decoder"] = checkpoint['decoder'].to(device).eval()
    global_models["resnet"]["encoder"] = checkpoint['encoder'].to(device).eval()

    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    global_models["resnet"]["word_map"] = word_map
    global_models["resnet"]["rev_word_map"] = {v: k for k, v in word_map.items()}
    print("resnet_lstm_attention 模型加载完成。")

def load_dino_model():
    """加载 dinov2_t5 模型"""
    if global_models["dino"]["captioner"] is not None:
        return

    print("正在加载 dinov2_t5 模型...")
    if DIR_DINO_SRC not in sys.path:
        sys.path.append(DIR_DINO_SRC)
    
    try:
        import config
        # 动态注入路径 (覆盖 config.py 中的默认值)
        config.VISION_MODEL_PATH = CONFIG['DINO_Model']['vision_model']
        config.TEXT_MODEL_PATH = CONFIG['DINO_Model']['text_model']
        
        from inference import ImageCaptioner
        
        lora_path = CONFIG['DINO_Model']['lora_weights']
        if not os.path.exists(lora_path):
             raise FileNotFoundError(f"找不到 LoRA 权重文件夹: {lora_path}")

        global_models["dino"]["captioner"] = ImageCaptioner(
            checkpoint_path=lora_path, 
            device=str(device)
        )
        print("dinov2_t5 模型加载完成。")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"DINOv2_T5 模型加载失败: {e}")

# ================= 4. 推理逻辑 =================

def run_inference(image_path, model_choice, beam_size=3):
    if image_path is None:
        return "<p style='text-align:center; color:#e74c3c;'>Please upload an image first.</p>"

    # 增强样式化的 HTML 模板：支持垂直滚动以应对长文本
    style_wrapper = """
    <div style='
        font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif; 
        font-size: 24px; 
        font-weight: bold; 
        color: #2C3E50; 
        text-align: center;
        padding: 30px 20px;
        margin-top: 10px;
        border: 3px double #3498db;
        border-radius: 15px;
        background-color: #f8fbfe;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        line-height: 1.6;
        min-height: 580px;
        max-height: 650px;
        overflow-y: auto;
    '>
        {}
    </div>
    """

    try:
        # 选项1: dinov2_t5 (匹配字符改为小写提高健壮性)
        if "dinov2" in model_choice.lower():
            load_dino_model()
            captioner = global_models["dino"]["captioner"]
            
            caption = captioner.generate_caption(
                image_path=image_path,
                num_beams=int(beam_size),
                return_scores=False 
            )
            return style_wrapper.format(caption)

        # 选项2: resnet_lstm_attention (关键点：改为匹配 lowercase)
        elif "resnet" in model_choice.lower():
            load_resnet_model()
            if DIR_RESNET not in sys.path:
                sys.path.append(DIR_RESNET)
            import caption as caption_module

            encoder = global_models["resnet"]["encoder"]
            decoder = global_models["resnet"]["decoder"]
            word_map = global_models["resnet"]["word_map"]
            rev_word_map = global_models["resnet"]["rev_word_map"]

            seq, _ = caption_module.caption_image_beam_search(
                encoder, decoder, image_path, word_map, int(beam_size)
            )

            words = [rev_word_map[ind] for ind in seq]
            result_caption = []
            for w in words:
                if w in ['<start>', '<end>', '<pad>']: continue
                result_caption.append(w)
            
            final_text = " ".join(result_caption)
            return style_wrapper.format(final_text)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# ================= 5. Gradio 界面 (美化版) =================

# 自定义 CSS 提升界面质感
custom_css = """
.container { max-width: 1100px; margin: auto; padding-top: 20px; }
.gradio-container { background-color: #fcfcfc !important; }
.img-box { border-radius: 12px; overflow: hidden; box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
.submit-btn { background: linear-gradient(135deg, #3498db, #2980b9) !important; color: white !important; }
.title-text { text-align: center; color: #34495e; margin-bottom: 30px; }
"""

with gr.Blocks(title="服装图像描述生成系统", css=custom_css) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# 👔 服装图像描述生成系统", elem_classes="title-text")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🛠️ 任务配置 & 输入")
                gr.Markdown("### 通过先进的深度学习模型自动生成细致的服装描述")
                
                image_input = gr.Image(
                    type="filepath", 
                    label="上传图片", 
                    height=420,
                    sources=["upload"],
                    elem_classes="img-box"
                )
                
                with gr.Group():
                    gr.Markdown("### 配置参数")
                    model_selector = gr.Radio(
                        choices=["ViT+Transformer (dinov2_t5)", "CNN+RNN (resnet_lstm_attention)"],
                        value="DINOv2 (dinov2_t5)",
                        label="选择推理引擎"
                    )
                    beam_slider = gr.Slider(
                        minimum=1, maximum=10, step=1, value=3, 
                        label="Beam Size (推荐使用 3-5)"
                    )
                
                submit_btn = gr.Button("🚀 点击生成详细描述", variant="primary", size="lg", elem_classes="submit-btn")

            with gr.Column(scale=1):
                gr.Markdown("## 📄 查看结果")
                output_text = gr.HTML(
                    label="生成结果",
                    value="""
                    <div style='
                        height: 580px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center; 
                        border: 2px dashed #bdc3c7; 
                        border-radius: 15px;
                        color: #7f8c8d;
                        background: #fdfdfd;
                        overflow-y: auto;
                    '>
                        等待生成结果...
                    </div>
                    """
                )
                
                with gr.Accordion("项目说明", open=True):
                    gr.Markdown("""
                    - **DINOv2-T5 (推荐)**: 采用 Meta DINOv2 视觉编码器与 Google Flan-T5 语言解码器。结合 **SwiGLU** 映射层与 **VC-BDR** 多样性损失函数，生成的描述详尽且极具语义丰富度。
                    - **ResNet-Attention (基准)**: 经典 CNN+RNN 架构，配合自适应空间注意力机制，推理响应极快，适合对生成速度要求较高的场景。
                    - **技术保障**: 模型基于 **DeepFashion** 数据集训练，并利用 **LCM-LoRA** 进行了实景背景合成增强，确保在真实社交与街拍环境下的描述准确性。
                    """)

    submit_btn.click(
        fn=run_inference,
        inputs=[image_input, model_selector, beam_slider],
        outputs=output_text
    )

if __name__ == "__main__":
    print(f"当前工作目录: {os.getcwd()}")
    print("启动 Gradio 服务...")
    demo.launch(inbrowser=True)