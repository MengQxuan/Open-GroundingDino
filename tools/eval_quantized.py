import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

"""
Open-GroundingDINO INT8 量化模型 COCO mAP 评估

用法：
    # 快速验证（50张，约10分钟）
    python tools/eval_quantized.py \
        --config_file outputs/xxx/config_cfg.py \
        --checkpoint_path outputs/xxx/checkpoint_best_regular.pth \
        --datasets config/datasets_coco_10k1k.json \
        --num_samples 50 --no_eval_fp32

    # 完整评估（1000张，约3小时）
    python tools/eval_quantized.py \
        --config_file outputs/xxx/config_cfg.py \
        --checkpoint_path outputs/xxx/checkpoint_best_regular.pth \
        --datasets config/datasets_coco_10k1k.json
"""
import argparse
import copy
import time
import json
import tempfile

import torch
import torch.quantization
from torch.utils.data import DataLoader, Subset

from util.slconfig import DictAction, SLConfig
import util.misc as utils

from datasets import build_dataset, get_coco_api_from_dataset
from datasets.cocogrounding_eval import CocoGroundingEvaluator
from groundingdino.util.utils import clean_state_dict


def _ensure_cfg_defaults(cfg):
    """补全 main.py 通过 argparse 设置、但 config_cfg.py 中没有的字段"""
    defaults = {
        "fix_size": False,
        "remove_difficult": False,
        "debug": False,
        "amp": False,
        "eval": True,
        "distributed": False,
        "save_results": False,
        "save_log": False,
        "useCats": True,
    }
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)


def build_full_model(args):
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = clean_state_dict(checkpoint["model"])
    else:
        state_dict = clean_state_dict(checkpoint)
    load_res = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] 权重加载: {load_res}")
    return model


def quantize_model(model):
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def get_model_size_mb(model):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / (1024 * 1024)


def to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    return v


def format_eta(seconds):
    """把秒数格式化成 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


@torch.no_grad()
def evaluate_model(model, criterion, postprocessors, data_loader, base_ds,
                   device, args, tag="", total_images=0):
    model.eval()
    criterion.eval()

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=True)

    if getattr(args, 'use_coco_eval', False) and getattr(args, 'coco_val_path', None):
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list = args.label_list
    caption = " . ".join(cat_list) + ' .'
    print(f"   [{tag}] text prompt: {caption[:80]}...")
    print(f"   [{tag}] 开始评估 {total_images} 张图片...")

    total_time = 0.0
    n_images = 0
    eval_start = time.perf_counter()

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs

        t0 = time.perf_counter()
        outputs = model(samples, captions=input_captions)
        total_time += time.perf_counter() - t0
        n_images += bs

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

        # 每 10 张打印进度 + 预计剩余时间
        if n_images % 10 == 0 or n_images == total_images:
            avg_ms = total_time / n_images * 1000
            elapsed = time.perf_counter() - eval_start
            remaining = (total_images - n_images) * (elapsed / n_images) if n_images > 0 else 0
            eta = format_eta(remaining)
            print(f"   [{tag}] {n_images}/{total_images} | "
                  f"avg {avg_ms:.0f} ms/img | "
                  f"ETA: {eta}")

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    avg_time = total_time / max(n_images, 1)
    print(f"   [{tag}] 完成! {n_images} 张, "
          f"avg {avg_time*1000:.1f} ms/img, {1.0/max(avg_time,1e-6):.1f} FPS")

    stats = {}
    for iou_type in iou_types:
        stats[f"{iou_type}_mAP"] = coco_evaluator.coco_eval[iou_type].stats[0]
        stats[f"{iou_type}_mAP50"] = coco_evaluator.coco_eval[iou_type].stats[1]
        stats[f"{iou_type}_mAP75"] = coco_evaluator.coco_eval[iou_type].stats[2]

    stats["avg_time_ms"] = avg_time * 1000
    stats["fps"] = 1.0 / max(avg_time, 1e-6)

    return stats


def main():
    parser = argparse.ArgumentParser("INT8 量化模型 COCO mAP 对比评估")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True,
                        help="数据集 JSON, e.g. config/datasets_coco_10k1k.json")
    parser.add_argument("--options", nargs="+", action=DictAction, default={})
    parser.add_argument("--device", default="cpu",
                        help="评估设备（INT8 仅支持 CPU）")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_samples", default=-1, type=int,
                        help="只评估前 N 张图片（-1 = 全部，推荐 50-100 做快速验证）")
    parser.add_argument("--no_eval_fp32", action="store_true",
                        help="跳过 FP32 评估，只评估 INT8")
    args = parser.parse_args()

    # ===== 加载配置 =====
    cfg = SLConfig.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
    cfg.device = args.device
    _ensure_cfg_defaults(cfg)

    # ===== 加载数据集 JSON =====
    with open(args.datasets, 'r') as f:
        dataset_meta = json.load(f)

    val_info = dataset_meta["val"][0]
    if val_info.get("anno"):
        cfg.coco_val_path = val_info["anno"]

    # ===== 构建验证数据集 =====
    print("=" * 60)
    print("📁 构建验证数据集...")
    print("=" * 60)
    dataset_val = build_dataset(image_set='val', args=cfg, datasetinfo=val_info)

    # 如果指定了 num_samples，只取前 N 张
    total_images = len(dataset_val)
    if args.num_samples > 0 and args.num_samples < total_images:
        indices = list(range(args.num_samples))
        dataset_val_subset = Subset(dataset_val, indices)
        total_images = args.num_samples
        print(f"   ⚡ 快速模式: 只评估前 {total_images} 张 (共 {len(dataset_val)} 张)")
    else:
        dataset_val_subset = dataset_val
        print(f"   验证集大小: {total_images}")

    # 预估时间
    est_per_img_s = 12.0  # 根据之前测试约 12s/img
    est_total_s = total_images * est_per_img_s
    print(f"   ⏱️  预计 INT8 评估时间: ~{format_eta(est_total_s)}")

    sampler_val = torch.utils.data.SequentialSampler(dataset_val_subset)
    data_loader_val = DataLoader(
        dataset_val_subset,
        batch_size=1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    # base_ds 必须用完整数据集（COCOeval 需要完整标注）
    base_ds = get_coco_api_from_dataset(dataset_val)

    # ===== 构建模型 =====
    print("\n" + "=" * 60)
    print("🔧 构建模型...")
    print("=" * 60)
    model, criterion, postprocessors = build_full_model(cfg)
    model = load_checkpoint(model, args.checkpoint_path)

    total_params = sum(p.numel() for p in model.parameters())
    fp32_size = get_model_size_mb(model)
    print(f"   参数量: {total_params / 1e6:.1f}M")
    print(f"   FP32 模型大小: {fp32_size:.1f} MB")

    device = torch.device(args.device)

    # ===== 评估 FP32 基准（可选）=====
    results_fp32 = None
    if not args.no_eval_fp32:
        print("\n" + "=" * 60)
        print("📊 [1/2] 评估 FP32 原始模型...")
        print("=" * 60)
        model_fp32 = copy.deepcopy(model)
        model_fp32.to(device)
        criterion.to(device)
        for k in postprocessors:
            postprocessors[k].to(device)

        results_fp32 = evaluate_model(
            model_fp32, criterion, postprocessors,
            data_loader_val, base_ds, device, cfg,
            tag="FP32", total_images=total_images
        )
        del model_fp32

    # ===== 量化 + 评估 INT8 =====
    print("\n" + "=" * 60)
    step = "2/2" if not args.no_eval_fp32 else "1/1"
    print(f"📊 [{step}] 量化并评估 INT8 模型...")
    print("=" * 60)

    model_int8 = quantize_model(model)
    int8_size = get_model_size_mb(model_int8)
    print(f"   INT8 模型大小: {int8_size:.1f} MB")
    print(f"   压缩比: {fp32_size / int8_size:.2f}x, 体积减少 {(1 - int8_size / fp32_size) * 100:.1f}%")

    cpu_device = torch.device("cpu")
    model_int8.to(cpu_device)
    criterion_cpu = copy.deepcopy(criterion).to(cpu_device)

    postprocessors_cpu = {}
    for k, v in postprocessors.items():
        postprocessors_cpu[k] = copy.deepcopy(v).to(cpu_device)

    results_int8 = evaluate_model(
        model_int8, criterion_cpu, postprocessors_cpu,
        data_loader_val, base_ds, cpu_device, cfg,
        tag="INT8", total_images=total_images
    )

    # ===== 汇总对比 =====
    print("\n" + "=" * 60)
    print("📋 最终对比结果")
    if args.num_samples > 0:
        print(f"   (基于前 {total_images} 张图片的评估)")
    print("=" * 60)

    header = f"  {'指标':<20} {'FP32':<12} {'INT8':<12} {'差异':<12}"
    print(header)
    print("  " + "-" * 56)

    for key in ["bbox_mAP", "bbox_mAP50", "bbox_mAP75", "avg_time_ms", "fps"]:
        v2 = results_int8.get(key, 0)
        if results_fp32:
            v1 = results_fp32.get(key, 0)
            diff = v2 - v1
            if "mAP" in key:
                print(f"  {key:<20} {v1:<12.4f} {v2:<12.4f} {diff:+.4f}")
            else:
                print(f"  {key:<20} {v1:<12.1f} {v2:<12.1f} {diff:+.1f}")
        else:
            if "mAP" in key:
                print(f"  {key:<20} {'(已知0.514)':<12} {v2:<12.4f}")
            else:
                print(f"  {key:<20} {'--':<12} {v2:<12.1f}")

    print("  " + "-" * 56)
    print(f"  {'模型体积(MB)':<20} {fp32_size:<12.1f} {int8_size:<12.1f} "
          f"-{(1 - int8_size / fp32_size) * 100:.1f}%")
    print("=" * 60)
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()