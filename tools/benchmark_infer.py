import os
import json
import time
import statistics
import argparse

import torch

# 复用 main.py 的完整 parser + build_model_main
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util import misc as utils
from datasets import build_dataset


def _cfg_to_args(args):
    """
    规则：args(命令行)优先；cfg 只补全 args 里没有的字段
    冲突字段直接跳过（不 raise），避免 num_queries / batch_size 等常见字段冲突
    """
    cfg = SLConfig.fromfile(args.config_file)
    if getattr(args, "options", None):
        cfg.merge_from_dict(args.options)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)

    for k, v in cfg_dict.items():
        # 如果 args 里已经有这个字段（无论默认值还是用户传的），都以 args 为准
        if k in args_vars:
            continue
        setattr(args, k, v)

    if not getattr(args, "debug", None):
        args.debug = False
    return args



def _load_dataset_meta(datasets_json_path):
    with open(datasets_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def _build_val_loader(args, dataset_meta):
    val_info = dataset_meta["val"][0]
    dataset_val = build_dataset(image_set="val", args=args, datasetinfo=val_info)

    sampler = torch.utils.data.SequentialSampler(dataset_val)
    loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=utils.collate_fn,
    )
    return loader


def _ensure_coco_val_path(args, dataset_meta):
    if getattr(args, "coco_val_path", None):
        return

    default_full = "data/coco/annotations/instances_val2017.json"
    if os.path.exists(default_full):
        args.coco_val_path = default_full
        return

    try:
        args.coco_val_path = dataset_meta["val"][0]["anno_path"]
    except Exception:
        pass


def _to_device(samples, targets, device):
    # samples 是 NestedTensor（utils.misc.NestedTensor）
    samples = samples.to(device)
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    return samples, targets


@torch.no_grad()
def main():
    parser = get_args_parser()

    # ===== benchmark 额外参数 =====
    parser.add_argument("--warmup", type=int, default=50, help="warmup iters (not timed)")
    parser.add_argument("--iters", type=int, default=200, help="timed iters")
    parser.add_argument("--num_queries", type=int, default=None, help="override num_queries for speed test")
    parser.add_argument("--caption", type=str, default="person .", help="non-empty caption")
    parser.add_argument("--forward_only", action="store_true", help="only measure model forward (no postprocess)")
    # =============================

    args = parser.parse_args()

    # cfg 注入
    args = _cfg_to_args(args)

    # datasets meta
    dataset_meta = _load_dataset_meta(args.datasets)
    _ensure_coco_val_path(args, dataset_meta)

    # num_queries 只能走 args（不能放 options）
    if args.num_queries is not None:
        # main parser 里本来就有 num_queries，这里只是确保你命令行传入生效
        pass

    if not getattr(args, "output_dir", None):
        args.output_dir = "./outputs/benchmark_tmp"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # build model
    model, _, postprocessors = build_model_main(args)
    model.to(device)
    model.eval()

    # build loader & 取一个 batch（不把 dataloader 时间算进推理）
    loader = _build_val_loader(args, dataset_meta)
    it = iter(loader)
    samples, targets = next(it)

    # 先搬到 device（不计入耗时）
    samples, targets = _to_device(samples, targets, device)

    # captions：固定 batch_size 个（避免每次循环构造列表的微小开销）
    # 注意：最后一个 batch 可能小于 batch_size，但我们这里固定只测这个 batch，因此直接按 samples.tensors.shape[0]
    bs = samples.tensors.shape[0]
    captions = [args.caption] * bs

    # AMP context
    use_amp = bool(getattr(args, "amp", False)) and (device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(device_type="cpu", enabled=False)

    # ===== Warmup =====
    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        with amp_ctx:
            outputs = model(samples, captions=captions)
            if not args.forward_only:
                _ = postprocessors["bbox"](outputs, targets)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ===== Timed iters =====
    times_ms = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(args.iters):
            starter.record()
            with amp_ctx:
                outputs = model(samples, captions=captions)
                if not args.forward_only:
                    _ = postprocessors["bbox"](outputs, targets)
            ender.record()
            ender.synchronize()
            times_ms.append(starter.elapsed_time(ender))

    else:
        # CPU：用 perf_counter（不建议用 CPU 测 GroundingDINO）
        for _ in range(args.iters):
            t0 = time.perf_counter()
            with amp_ctx:
                outputs = model(samples, captions=captions)
                if not args.forward_only:
                    _ = postprocessors["bbox"](outputs, targets)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    p50 = statistics.median(times_ms)
    p90 = sorted(times_ms)[int(0.9 * (len(times_ms) - 1))]

    # FPS 按 batch 计算：FPS = batch_size / latency_sec
    mean_fps = bs / (mean_ms / 1000.0)
    p50_fps = bs / (p50 / 1000.0)

    print("\n================ BENCHMARK ================")
    print(f"mode: {'forward_only' if args.forward_only else 'forward+postprocess'}")
    print(f"device: {args.device}")
    print(f"amp: {use_amp}")
    print(f"num_queries: {args.num_queries}")
    print(f"batch_size: {bs}, warmup: {args.warmup}, iters: {args.iters}")
    print(f"latency ms (mean/p50/p90): {mean_ms:.2f} / {p50:.2f} / {p90:.2f}")
    print(f"FPS (mean/p50): {mean_fps:.2f} / {p50_fps:.2f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
