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
    按 main.py 的逻辑：把 cfg 里“parser 没有定义的字段”注入到 args
    注意：main.py 对同名 key 会直接 raise，这里也保持一致
    """
    cfg = SLConfig.fromfile(args.config_file)
    if getattr(args, "options", None):
        # 你的 main.py 里 --options 是 dict，不是 list
        cfg.merge_from_dict(args.options)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError(f"Key {k} can used by args only")

    # main.py 里也会补 debug
    if not getattr(args, "debug", None):
        args.debug = False

    return args


def _load_dataset_meta(datasets_json_path):
    with open(datasets_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def _build_val_loader(args, dataset_meta):
    # 这个 repo 的 build_dataset 需要 datasetinfo
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
        collate_fn=utils.collate_fn,  # 复用项目的 collate
    )
    return loader


def _ensure_coco_val_path(args, dataset_meta):
    # 如果显式传了
    if getattr(args, "coco_val_path", None):
        return

    # 优先给一个稳定默认值
    default_full = "data/coco/annotations/instances_val2017.json"
    if os.path.exists(default_full):
        args.coco_val_path = default_full
        return

    # 兜底：用 datasets json 的 val anno_path
    try:
        args.coco_val_path = dataset_meta["val"][0]["anno_path"]
    except Exception:
        pass


@torch.no_grad()
def main():
    parser = get_args_parser()

    # ===== 只加 benchmark 需要的新参数（避免和 main parser 冲突）=====
    parser.add_argument("--warmup", type=int, default=50, help="warmup iters (not timed)")
    parser.add_argument("--iters", type=int, default=200, help="timed iters")
    parser.add_argument("--num_queries", type=int, default=None, help="override num_queries for speed test")
    parser.add_argument("--caption", type=str, default="person .", help="non-empty caption (must have at least one category token)")
    parser.add_argument("--forward_only", action="store_true", help="only measure model forward (no postprocess)")
    # ===============================================================

    args = parser.parse_args()

    # 让 cfg 把 hidden_dim/masks/fix_size/... 全部灌进 args
    args = _cfg_to_args(args)

    # datasets meta（用于 build_dataset & 某些路径）
    dataset_meta = _load_dataset_meta(args.datasets)

    # 覆盖 num_queries：用 options 的方式最保险（因为模型内部读取的是 args.num_queries）
    if args.num_queries is not None:
        args.num_queries = args.num_queries

    # 确保 coco_val_path 存在且 categories 完整（避免 KeyError: 'categories'）
    _ensure_coco_val_path(args, dataset_meta)

    # 重要：benchmark 不需要训练输出目录，但 build_model_main 可能依赖一些字段
    if not getattr(args, "output_dir", None):
        args.output_dir = "./outputs/benchmark_tmp"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # ===== build model =====
    model, _, postprocessors = build_model_main(args)
    model.to(device)
    model.eval()

    # ===== data loader =====
    loader = _build_val_loader(args, dataset_meta)
    it = iter(loader)

    # captions：必须非空，否则会遇到 stack non-empty TensorList
    cap = args.caption
    if not isinstance(cap, str) or len(cap.strip()) == 0:
        cap = "person ."

    # ===== warmup =====
    for _ in range(args.warmup):
        try:
            samples, targets = next(it)
        except StopIteration:
            it = iter(loader)
            samples, targets = next(it)

        samples = samples.to(device)
        captions = [cap] * len(targets)

        if args.amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(samples, captions=captions)
        else:
            outputs = model(samples, captions=captions)

        if not args.forward_only:
            # 端到端：包含后处理（会更慢，但更贴近真实推理 pipeline）
            _ = postprocessors["bbox"](outputs, torch.stack([t["orig_size"] for t in targets], dim=0))

    torch.cuda.synchronize()

    # ===== timed =====
    lat_ms = []
    for _ in range(args.iters):
        try:
            samples, targets = next(it)
        except StopIteration:
            it = iter(loader)
            samples, targets = next(it)

        samples = samples.to(device)
        captions = [cap] * len(targets)

        t0 = time.time()
        if args.amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(samples, captions=captions)
        else:
            outputs = model(samples, captions=captions)

        if not args.forward_only:
            _ = postprocessors["bbox"](outputs, torch.stack([t["orig_size"] for t in targets], dim=0))

        torch.cuda.synchronize()
        t1 = time.time()

        lat_ms.append((t1 - t0) * 1000.0)

    # ===== report =====
    bs = args.batch_size
    p50 = statistics.median(lat_ms)
    p90 = sorted(lat_ms)[int(0.9 * len(lat_ms))]
    mean = sum(lat_ms) / len(lat_ms)

    fps_mean = (1000.0 / mean) * bs
    fps_p50 = (1000.0 / p50) * bs

    mode = "forward_only" if args.forward_only else "e2e_with_postprocess"
    print("\n================ BENCHMARK ================")
    print(f"mode: {mode}")
    print(f"device: {args.device}")
    print(f"amp: {bool(args.amp)}")
    print(f"num_queries: {getattr(args, 'num_queries', None)}")
    print(f"batch_size: {bs}, warmup: {args.warmup}, iters: {args.iters}")
    print(f"latency ms (mean/p50/p90): {mean:.2f} / {p50:.2f} / {p90:.2f}")
    print(f"FPS (mean/p50): {fps_mean:.2f} / {fps_p50:.2f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
