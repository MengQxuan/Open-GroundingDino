import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

"""
Open-GroundingDINO INT8 动态量化推理脚本
用法：
python tools/quantize_dynamic.py \
    --config_file outputs/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32/config_cfg.py \
    --checkpoint_path outputs/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32/checkpoint_best_regular.pth \
    --image_path test.png \
    --text_prompt "person . dog ." \
    --output_dir quantitative_models/

python tools/quantize_dynamic.py \
    --config_file outputs/10k1k_q300/config_cfg.py \
    --checkpoint_path outputs/10k1k_q300/checkpoint_best_regular.pth \
    --image_path test.png \
    --text_prompt "person . dog ." \
    --output_dir quantitative_models/

输出文件名自动生成，格式为：
    {实验名}_{checkpoint名}_int8.pth
    例如：prune_enc4_dec3_q300_distilbert_np2_clip8_fp32_checkpoint_best_regular_int8.pth

python tools/quantize_dynamic.py \
    --config_file outputs/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32/config_cfg.py \
    --checkpoint_path outputs/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32/checkpoint_best_regular.pth \
    --image_path test.png \
    --text_prompt "person . dog ." \
    --output_path quantitative_models/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32_checkpoint_best_regular_int8 \
    --benchmark

python tools/quantize_dynamic.py \
    --config_file outputs/10k1k_q300/config_cfg.py \
    --checkpoint_path outputs/10k1k_q300/checkpoint_best_regular.pth \
    --image_path test.png \
    --text_prompt "person . dog ." \
    --output_path quantitative_models/10k1k_q300_checkpoint_best_regular_int8.pth \
    --benchmark
"""
import argparse
import time
import tempfile
from datetime import datetime

import torch
import torch.quantization
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


def auto_generate_output_path(checkpoint_path, output_dir):
    """
    根据 checkpoint 路径自动生成量化模型的输出路径
    
    示例：
      checkpoint_path = "outputs/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32/checkpoint_best_regular.pth"
      output_dir = "quantitative_models/"
      
      → "quantitative_models/prune_enc4_dec3_q300_distilbert_np2_clip8_fp32_checkpoint_best_regular_int8.pth"
    """
    # 提取实验名（checkpoint 所在目录名）
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    experiment_name = os.path.basename(ckpt_dir)
    
    # 提取 checkpoint 文件名（去掉 .pth 后缀）
    ckpt_filename = os.path.basename(checkpoint_path)
    ckpt_stem = os.path.splitext(ckpt_filename)[0]
    
    # 组合：{实验名}_{checkpoint名}_int8.pth
    output_filename = f"{experiment_name}_{ckpt_stem}_int8.pth"
    output_path = os.path.join(output_dir, output_filename)
    
    # 如果文件已存在，追加时间戳避免覆盖
    if os.path.exists(output_path):
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_filename = f"{experiment_name}_{ckpt_stem}_int8_{timestamp}.pth"
        output_path = os.path.join(output_dir, output_filename)
    
    return output_path


def load_image(image_path):
    """加载并预处理图像"""
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def build_and_load_model(config_path, checkpoint_path, device="cpu"):
    """构建并加载模型——仅加载 model 本体，跳过 criterion/postprocessors 的依赖"""
    args = SLConfig.fromfile(config_path)
    args.device = device

    # PostProcess.__init__ 在 use_coco_eval=True 时��访问 args.coco_val_path
    # 量化脚本不需要这些，临时关掉即可
    args.use_coco_eval = False
    if not hasattr(args, 'label_list') or getattr(args, 'label_list', None) is None:
        args.label_list = ["dummy"]

    model_result = build_model(args)
    if isinstance(model_result, tuple):
        model = model_result[0]
    else:
        model = model_result

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = clean_state_dict(checkpoint["model"])
    else:
        state_dict = clean_state_dict(checkpoint)

    load_res = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] 权重加载结果: {load_res}")
    model.eval()
    return model


def quantize_model_dynamic(model):
    """PyTorch 动态 INT8 量化，仅量化 nn.Linear 层"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def compare_model_size(original_model, quantized_model, save_path=None):
    """对比量化前后的模型大小"""
    with tempfile.NamedTemporaryFile(delete=True) as f:
        torch.save(original_model.state_dict(), f.name)
        original_size = os.path.getsize(f.name) / (1024 * 1024)

    with tempfile.NamedTemporaryFile(delete=True) as f:
        torch.save(quantized_model.state_dict(), f.name)
        quantized_size = os.path.getsize(f.name) / (1024 * 1024)

    print(f"\n{'='*50}")
    print(f"📦 模型大小对比:")
    print(f"   原始模型 (FP32):  {original_size:.1f} MB")
    print(f"   量化模型 (INT8):  {quantized_size:.1f} MB")
    print(f"   压缩比:           {original_size / quantized_size:.2f}x")
    print(f"   体积减少:         {(1 - quantized_size / original_size) * 100:.1f}%")
    print(f"{'='*50}\n")

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        torch.save(quantized_model.state_dict(), save_path)
        print(f"[INFO] 量化模型已保存到: {save_path}")

    return original_size, quantized_size


def benchmark_inference(model, image, caption, device="cpu", n_warmup=3, n_runs=10):
    """推理速度基准测试"""
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                _ = model(image[None], captions=[caption])
            except Exception as e:
                print(f"[WARN] 推理出错: {e}")
                return float('inf')

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(image[None], captions=[caption])
            end = time.perf_counter()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"   平均推理时间: {avg_time * 1000:.1f} ms  ({1.0 / avg_time:.1f} FPS)")
    return avg_time


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser("Open-GroundingDINO INT8 动态量化")
    parser.add_argument("--config_file", type=str, required=True,
                        help="配置文件路径, e.g. outputs/xxx/config_cfg.py")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--image_path", type=str, default="test.png",
                        help="测试图像路径")
    parser.add_argument("--text_prompt", type=str, default="person . car .",
                        help="文本提示")

    # ====== 输出路径：二选一 ======
    parser.add_argument("--output_path", type=str, default=None,
                        help="手动指定量化模型保存路径（优先级最高）")
    parser.add_argument("--output_dir", type=str, default="quantitative_models",
                        help="量化模型保存目录，文件名自动生成（默认: quantitative_models/）")
    # =============================

    parser.add_argument("--benchmark", action="store_true",
                        help="是否进行推理速度对比")
    args = parser.parse_args()

    # ====== 确定输出路径 ======
    if args.output_path:
        # 用户手动指定了完整路径，直接用
        output_path = args.output_path
    else:
        # 自动生成文件名
        output_path = auto_generate_output_path(args.checkpoint_path, args.output_dir)

    print(f"[INFO] 量化模型将保存到: {output_path}")

    # 1. 加载模型
    print("\n[Step 1] 加载原始模型...")
    model = build_and_load_model(args.config_file, args.checkpoint_path, device="cpu")

    total_params, _ = count_parameters(model)
    print(f"   总参数量: {total_params / 1e6:.1f}M")

    # 2. 动态量化
    print("\n[Step 2] 执行 INT8 动态量化...")
    quantized_model = quantize_model_dynamic(model)

    # 3. 对比模型大小
    print("\n[Step 3] 模型大小对比...")
    compare_model_size(model, quantized_model, save_path=output_path)

    # 4. 推理速度对比（可选）
    if args.benchmark and os.path.exists(args.image_path):
        print("\n[Step 4] 推理速度对比 (CPU)...")
        _, image = load_image(args.image_path)
        caption = args.text_prompt

        print("   原始模型:")
        benchmark_inference(model, image, caption, device="cpu")

        print("   量化模型:")
        benchmark_inference(quantized_model, image, caption, device="cpu")

    print("\n 量化完成！")


if __name__ == "__main__":
    main()