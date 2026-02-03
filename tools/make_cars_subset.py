import os, json, random
from pathlib import Path

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def symlink_or_copy(src: Path, dst: Path, mode: str):
    ensure(dst.parent)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(str(src), str(dst))
    elif mode == "copy":
        import shutil
        shutil.copy2(str(src), str(dst))
    else:
        raise ValueError("SUBSET_MODE must be symlink or copy")

def sample_train_jsonl(train_jsonl: Path, n_train: int, seed: int):
    lines = train_jsonl.read_text(encoding="utf-8").splitlines()
    if n_train > len(lines):
        raise ValueError(f"n_train({n_train}) > total train lines({len(lines)})")
    rnd = random.Random(seed)
    idx = list(range(len(lines)))
    rnd.shuffle(idx)
    chosen = idx[:n_train]
    sampled = [lines[i] for i in chosen]
    return sampled

def sample_val_coco(val_json: Path, n_val: int, seed: int):
    data = json.load(open(val_json, "r", encoding="utf-8"))
    images = data["images"]
    anns = data["annotations"]
    cats = data["categories"]

    if n_val > len(images):
        raise ValueError(f"n_val({n_val}) > total val images({len(images)})")

    imgid_to_anns = {}
    for a in anns:
        imgid_to_anns.setdefault(a["image_id"], []).append(a)

    rnd = random.Random(seed)
    ids = [im["id"] for im in images]
    rnd.shuffle(ids)
    chosen_ids = set(ids[:n_val])

    new_images = []
    new_anns = []
    oldid_to_newid = {}
    new_img_id = 1
    new_ann_id = 1

    for im in images:
        if im["id"] not in chosen_ids:
            continue
        oldid_to_newid[im["id"]] = new_img_id
        new_images.append({**im, "id": new_img_id})

        for a in imgid_to_anns.get(im["id"], []):
            na = dict(a)
            na["id"] = new_ann_id
            na["image_id"] = new_img_id
            new_anns.append(na)
            new_ann_id += 1
        new_img_id += 1

    return {"images": new_images, "annotations": new_anns, "categories": cats}

def main():
    repo = Path(__file__).resolve().parents[1]

    # 关键：允许通过 DS_DIR 指定数据集目录
    # 默认还是 stanford_cars
    ds_name = os.environ.get("DS_DIR", "stanford_cars").strip()
    ds = repo / "data" / ds_name
    ann_dir = ds / "annotations"

    train_jsonl = ann_dir / "train.jsonl"
    val_json = ann_dir / "val.json"
    label_map = ann_dir / "label_map.json"

    if not train_jsonl.exists():
        raise FileNotFoundError(f"Missing {train_jsonl}")
    if not val_json.exists():
        raise FileNotFoundError(f"Missing {val_json}")
    if not label_map.exists():
        raise FileNotFoundError(f"Missing {label_map}")

    n_train = int(os.environ.get("N_TRAIN", "1000"))
    n_val = int(os.environ.get("N_VAL", "100"))
    seed = int(os.environ.get("SEED", "42"))
    mode = os.environ.get("SUBSET_MODE", "symlink").strip().lower()

    out_train_jsonl = ann_dir / f"train_{n_train}.jsonl"
    out_val_json = ann_dir / f"val_{n_val}.json"

    out_train_img_dir = ds / f"train_images_{n_train}"
    out_val_img_dir = ds / f"val_images_{n_val}"

    ensure(out_train_img_dir)
    ensure(out_val_img_dir)

    # 说明：图片源目录沿用原 stanford_cars 的 split（最省空间）
    # 如果 ds_name 是 stanford_cars_20cls，我们默认图片在 data/stanford_cars/train_images & val_images
    if ds_name == "stanford_cars_20cls":
        img_src_root = repo / "data" / "stanford_cars"
    else:
        img_src_root = ds

    src_train_img_dir = img_src_root / "train_images"
    src_val_img_dir = img_src_root / "val_images"

    # ---- train subset ----
    sampled_lines = sample_train_jsonl(train_jsonl, n_train=n_train, seed=seed)
    out_train_jsonl.write_text("\n".join(sampled_lines) + "\n", encoding="utf-8")

    for line in sampled_lines:
        obj = json.loads(line)
        fn = Path(obj["filename"]).name
        src = src_train_img_dir / fn
        if not src.exists():
            raise FileNotFoundError(f"train image missing: {src}")
        dst = out_train_img_dir / fn
        symlink_or_copy(src, dst, mode=mode)

    # ---- val subset ----
    sampled_val = sample_val_coco(val_json, n_val=n_val, seed=seed)
    out_val_json.write_text(json.dumps(sampled_val, ensure_ascii=False), encoding="utf-8")

    for im in sampled_val["images"]:
        fn = Path(im["file_name"]).name
        src = src_val_img_dir / fn
        if not src.exists():
            raise FileNotFoundError(f"val image missing: {src}")
        dst = out_val_img_dir / fn
        symlink_or_copy(src, dst, mode=mode)

    # ---- dataset cfg json (new) ----
    out_cfg = repo / "config" / f"datasets_{ds_name}_odvg_{n_train}_{n_val}.json"
    cfg = {
        "train": [{
            "root": str(out_train_img_dir),
            "anno": str(out_train_jsonl),
            "label_map": str(label_map),
            "dataset_mode": "odvg"
        }],
        "val": [{
            "root": str(out_val_img_dir),
            "anno": str(out_val_json),
            "label_map": None,
            "dataset_mode": "coco"
        }]
    }
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), encoding="utf-8")

    print("[DONE]")
    print("DS_DIR:", ds_name)
    print("train jsonl:", out_train_jsonl)
    print("val json:", out_val_json)
    print("train imgs:", out_train_img_dir, "count=", len(list(out_train_img_dir.iterdir())))
    print("val imgs:", out_val_img_dir, "count=", len(list(out_val_img_dir.iterdir())))
    print("dataset cfg:", out_cfg)

if __name__ == "__main__":
    main()