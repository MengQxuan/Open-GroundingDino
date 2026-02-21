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


def _build_caption_from_len(caption_len: int, token: str = "a", suffix: str = ".") -> str:
    caption_len = int(caption_len)
    if caption_len <= 0:
        return "person ."
    cap = ("{} " * caption_len).format(*([token] * caption_len)).strip()
    return cap + suffix


def _load_caption_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    return "person ."


@torch.no_grad()
def main():
    parser = get_args_parser()

    # ===== benchmark 额外参数 =====
    parser.add_argument("--warmup", type=int, default=50, help="warmup iters (not timed)")
    parser.add_argument("--iters", type=int, default=200, help="timed iters")
    parser.add_argument("--num_queries", type=int, default=None, help="override num_queries for speed test")
    # ===== [Stage5] split profile & quant =====
    parser.add_argument("--profile_split", action="store_true", help="split timing into tokenize/text/vision")
    parser.add_argument("--text_device", type=str, default=None, choices=["cpu", "cuda"], help="run text encoder on cpu/cuda")
    parser.add_argument("--dynamic_int8_text", action="store_true", help="dynamic int8 for text tower (CPU only)")

    # caption 控制（新增）
    parser.add_argument("--caption", type=str, default="person .", help="caption text (used if caption_len/caption_file not set)")
    parser.add_argument("--caption_len", type=int, default=None, help="if set, generate caption with N repeated tokens (e.g., 'a a a ...').")
    parser.add_argument("--caption_token", type=str, default="a", help="token used when caption_len is set")
    parser.add_argument("--caption_suffix", type=str, default=".", help="suffix appended to generated caption")
    parser.add_argument("--caption_file", type=str, default=None, help="if set, load the first non-empty line as caption")

    parser.add_argument("--forward_only", action="store_true", help="only measure model forward (no postprocess)")

    # 多 batch 平均（新增）
    parser.add_argument("--num_batches", type=int, default=1,
                        help="number of distinct batches to prefetch to GPU and rotate during timing (reduces outlier bias)")
    # =============================

    args = parser.parse_args()

    # cfg 注入
    args = _cfg_to_args(args)

    # datasets meta
    dataset_meta = _load_dataset_meta(args.datasets)
    _ensure_coco_val_path(args, dataset_meta)

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
    print("[DBG] text model class =", type(model.bert.text_encoder if hasattr(model.bert, "text_encoder") else model.bert))
    
    # ===== [Stage5-1] dynamic quant (CPU only) =====
    if args.dynamic_int8_text:
        import torch.ao.quantization as aq
        print("[Stage5] enable dynamic int8 quant for text tower (CPU only)")

        # 你这个 repo 的 TextEncoderShell 里是 self.text_model
        if hasattr(model.bert, "text_model"):
            model.bert.text_model = model.bert.text_model.cpu()
            model.bert.text_model = aq.quantize_dynamic(
                model.bert.text_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            # 强制 text_device=cpu
            if args.text_device is None:
                args.text_device = "cpu"
        else:
            print("[WARN] model.bert has no attribute text_model, skip dynamic quant.")

    # build loader & 预取若干 batch（不把 dataloader / H2D 拷贝时间算进推理）
    loader = _build_val_loader(args, dataset_meta)
    it = iter(loader)

    prefetched = []
    target_bs = None
    nb = max(1, int(args.num_batches))
    for _ in range(nb):
        try:
            samples, targets = next(it)
        except StopIteration:
            break
        samples, targets = _to_device(samples, targets, device)
        bs = samples.tensors.shape[0]
        if target_bs is None:
            target_bs = bs
        if bs != target_bs:
            continue
        prefetched.append((samples, targets))

    if len(prefetched) == 0:
        raise RuntimeError("No batches prefetched. Check dataset/loader settings.")

    bs = target_bs

    # captions：固定 batch_size 个（避免每次循环构造列表的微小开销）
    if args.caption_file:
        cap = _load_caption_file(args.caption_file)
    elif args.caption_len is not None:
        cap = _build_caption_from_len(args.caption_len, token=args.caption_token, suffix=args.caption_suffix)
    else:
        cap = args.caption

    captions = [cap] * bs

    # AMP context
    use_amp = bool(getattr(args, "amp", False)) and (device.type == "cuda")
    if use_amp:
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_ctx = torch.autocast(device_type="cpu", enabled=False)

    # ===== Warmup =====
    if device.type == "cuda":
        torch.cuda.synchronize()

    for i in range(args.warmup):
        samples, targets = prefetched[i % len(prefetched)]
        with amp_ctx:
            # outputs = model(samples, captions=captions)
            forward_kw = {"captions": captions}
            if args.profile_split:
                forward_kw["profile_split"] = True
                if args.text_device is not None:
                    forward_kw["text_device"] = torch.device(args.text_device)
            outputs = model(samples, **forward_kw)
            if not args.forward_only:
                _ = postprocessors["bbox"](outputs, targets)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ===== Timed iters =====
    times_ms = []
    t_tokenize_ms = []
    t_text_ms = []
    t_vision_ms = []
    t_backbone_ms = []
    t_transformer_ms = []
    t_encoder_ms = []
    t_decoder_ms = []
    t_enc_fusion_ms = []
    t_enc_text_ms = []
    t_enc_msdeform_ms = []
    t_enc_fusion_ln_ms = []
    t_enc_fusion_attn_ms = []
    t_enc_fusion_resid_ms = []
    t_enc_fusion_proj_ms = []
    t_enc_fusion_scores_ms = []
    t_enc_fusion_softmax_ms = []
    t_enc_fusion_ctx_ms = []
    t_enc_fusion_out_ms = []
    t_heads_ms = []

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for i in range(args.iters):
            samples, targets = prefetched[i % len(prefetched)]
            starter.record()
            with amp_ctx:
                # outputs = model(samples, captions=captions)
                forward_kw = {"captions": captions}
                if args.profile_split:
                    forward_kw["profile_split"] = True
                    if args.text_device is not None:
                        forward_kw["text_device"] = torch.device(args.text_device)
                outputs = model(samples, **forward_kw)
                if not args.forward_only:
                    _ = postprocessors["bbox"](outputs, targets)
            ender.record()
            ender.synchronize()
            times_ms.append(starter.elapsed_time(ender))
            if args.profile_split:
                p = getattr(model, "last_profile", None)
                if p is not None:
                    t_tokenize_ms.append(p.get("T_tokenize_ms", 0.0))
                    t_text_ms.append(p.get("T_text_encoder_ms", 0.0))
                    t_vision_ms.append(p.get("T_vision_decoder_ms", 0.0))
                    t_backbone_ms.append(p.get("T_backbone_ms", 0.0))
                    t_transformer_ms.append(p.get("T_transformer_ms", 0.0))
                    t_encoder_ms.append(p.get("T_encoder_ms", 0.0))
                    t_decoder_ms.append(p.get("T_decoder_ms", 0.0))
                    t_enc_fusion_ms.append(p.get("T_enc_fusion_ms", 0.0))
                    t_enc_text_ms.append(p.get("T_enc_text_ms", 0.0))
                    t_enc_msdeform_ms.append(p.get("T_enc_msdeform_ms", 0.0))
                    t_enc_fusion_ln_ms.append(p.get("T_enc_fusion_ln_ms", 0.0))
                    t_enc_fusion_attn_ms.append(p.get("T_enc_fusion_attn_ms", 0.0))
                    t_enc_fusion_resid_ms.append(p.get("T_enc_fusion_resid_ms", 0.0))
                    t_enc_fusion_proj_ms.append(p.get("T_enc_fusion_proj_ms", 0.0))
                    t_enc_fusion_scores_ms.append(p.get("T_enc_fusion_scores_ms", 0.0))
                    t_enc_fusion_softmax_ms.append(p.get("T_enc_fusion_softmax_ms", 0.0))
                    t_enc_fusion_ctx_ms.append(p.get("T_enc_fusion_ctx_ms", 0.0))
                    t_enc_fusion_out_ms.append(p.get("T_enc_fusion_out_ms", 0.0))
                    t_heads_ms.append(p.get("T_heads_ms", 0.0))
    else:
        for i in range(args.iters):
            samples, targets = prefetched[i % len(prefetched)]
            t0 = time.perf_counter()
            with amp_ctx:
                # outputs = model(samples, captions=captions)
                forward_kw = {"captions": captions}
                if args.profile_split:
                    forward_kw["profile_split"] = True
                    if args.text_device is not None:
                        forward_kw["text_device"] = torch.device(args.text_device)
                outputs = model(samples, **forward_kw)
                if not args.forward_only:
                    _ = postprocessors["bbox"](outputs, targets)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
            if args.profile_split:
                p = getattr(model, "last_profile", None)
                if p is not None:
                    t_tokenize_ms.append(p.get("T_tokenize_ms", 0.0))
                    t_text_ms.append(p.get("T_text_encoder_ms", 0.0))
                    t_vision_ms.append(p.get("T_vision_decoder_ms", 0.0))
                    t_backbone_ms.append(p.get("T_backbone_ms", 0.0))
                    t_transformer_ms.append(p.get("T_transformer_ms", 0.0))
                    t_encoder_ms.append(p.get("T_encoder_ms", 0.0))
                    t_decoder_ms.append(p.get("T_decoder_ms", 0.0))
                    t_enc_fusion_ms.append(p.get("T_enc_fusion_ms", 0.0))
                    t_enc_text_ms.append(p.get("T_enc_text_ms", 0.0))
                    t_enc_msdeform_ms.append(p.get("T_enc_msdeform_ms", 0.0))
                    t_enc_fusion_ln_ms.append(p.get("T_enc_fusion_ln_ms", 0.0))
                    t_enc_fusion_attn_ms.append(p.get("T_enc_fusion_attn_ms", 0.0))
                    t_enc_fusion_resid_ms.append(p.get("T_enc_fusion_resid_ms", 0.0))
                    t_enc_fusion_proj_ms.append(p.get("T_enc_fusion_proj_ms", 0.0))
                    t_enc_fusion_scores_ms.append(p.get("T_enc_fusion_scores_ms", 0.0))
                    t_enc_fusion_softmax_ms.append(p.get("T_enc_fusion_softmax_ms", 0.0))
                    t_enc_fusion_ctx_ms.append(p.get("T_enc_fusion_ctx_ms", 0.0))
                    t_enc_fusion_out_ms.append(p.get("T_enc_fusion_out_ms", 0.0))
                    t_heads_ms.append(p.get("T_heads_ms", 0.0))

    mean_ms = sum(times_ms) / len(times_ms)
    p50 = statistics.median(times_ms)
    p90 = sorted(times_ms)[int(0.9 * (len(times_ms) - 1))]

    mean_fps = bs / (mean_ms / 1000.0)
    p50_fps = bs / (p50 / 1000.0)

    print("\n================ BENCHMARK ================")
    print(f"mode: {'forward_only' if args.forward_only else 'forward+postprocess'}")
    print(f"device: {args.device}")
    print(f"amp: {use_amp}")
    print(f"num_queries: {args.num_queries}")
    print(f"batch_size: {bs}, warmup: {args.warmup}, iters: {args.iters}, num_batches: {len(prefetched)}")
    print(f"caption: {cap!r}")
    if args.caption_len is not None:
        print(f"caption_len: {args.caption_len} (token={args.caption_token!r})")
    print(f"latency ms (mean/p50/p90): {mean_ms:.2f} / {p50:.2f} / {p90:.2f}")
    print(f"FPS (mean/p50): {mean_fps:.2f} / {p50_fps:.2f}")
    if args.profile_split and len(t_text_ms) > 0:
        def _mean(x): return sum(x) / len(x)
        def _p50(x): return statistics.median(x)
        print("---- split(ms) mean/p50 ----")
        print(f"tokenize : {_mean(t_tokenize_ms):.3f} / {_p50(t_tokenize_ms):.3f}")
        print(f"text_enc : {_mean(t_text_ms):.3f} / {_p50(t_text_ms):.3f}")
        print(f"vision+dec: {_mean(t_vision_ms):.3f} / {_p50(t_vision_ms):.3f}")
        print(f"backbone : {_mean(t_backbone_ms):.3f} / {_p50(t_backbone_ms):.3f}")
        print(f"transfmr : {_mean(t_transformer_ms):.3f} / {_p50(t_transformer_ms):.3f}")
        if len(t_encoder_ms) > 0 and len(t_decoder_ms) > 0:
            print(f"  enc   : {_mean(t_encoder_ms):.3f} / {_p50(t_encoder_ms):.3f}")
            print(f"  dec   : {_mean(t_decoder_ms):.3f} / {_p50(t_decoder_ms):.3f}")
        if len(t_enc_fusion_ms) > 0:
            print(f"    enc_fusion : {_mean(t_enc_fusion_ms):.3f} / {_p50(t_enc_fusion_ms):.3f}")
            print(f"    enc_text   : {_mean(t_enc_text_ms):.3f} / {_p50(t_enc_text_ms):.3f}")
            print(f"    enc_msdef  : {_mean(t_enc_msdeform_ms):.3f} / {_p50(t_enc_msdeform_ms):.3f}")
            print(f"      fusion_ln : {_mean(t_enc_fusion_ln_ms):.3f} / {_p50(t_enc_fusion_ln_ms):.3f}")
            print(f"      fusion_attn: {_mean(t_enc_fusion_attn_ms):.3f} / {_p50(t_enc_fusion_attn_ms):.3f}")
            print(f"      fusion_resid: {_mean(t_enc_fusion_resid_ms):.3f} / {_p50(t_enc_fusion_resid_ms):.3f}")

            print(f"      attn_proj : {_mean(t_enc_fusion_proj_ms):.3f} / {_p50(t_enc_fusion_proj_ms):.3f}")
            print(f"      attn_scores: {_mean(t_enc_fusion_scores_ms):.3f} / {_p50(t_enc_fusion_scores_ms):.3f}")
            print(f"      attn_softmax: {_mean(t_enc_fusion_softmax_ms):.3f} / {_p50(t_enc_fusion_softmax_ms):.3f}")
            print(f"      attn_ctx  : {_mean(t_enc_fusion_ctx_ms):.3f} / {_p50(t_enc_fusion_ctx_ms):.3f}")
            print(f"      attn_out  : {_mean(t_enc_fusion_out_ms):.3f} / {_p50(t_enc_fusion_out_ms):.3f}")
        print(f"heads    : {_mean(t_heads_ms):.3f} / {_p50(t_heads_ms):.3f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
