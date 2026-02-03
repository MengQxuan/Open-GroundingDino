import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

from PIL import Image
from tqdm import tqdm
import scipy.io as sio


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_mat_annotations(mat_path: Path):
    """
    兼容 Stanford Cars 的 cars_annos.mat 字段:
      - relative_im_path (常见) 或 fname(少数版本)
      - bbox_x1 bbox_y1 bbox_x2 bbox_y2
      - class
      - test (0/1)
    返回: List[(rel_path, (x1,y1,x2,y2), cls1, test_flag)]
    """
    mat = sio.loadmat(str(mat_path))
    if "annotations" not in mat:
        raise KeyError(f"'annotations' not found in {mat_path}. keys={list(mat.keys())}")

    annos = mat["annotations"]
    # 常见形状 (1, N)
    if annos.ndim == 2:
        annos = annos[0]
    else:
        annos = annos.squeeze()

    # 判断字段名
    names = getattr(annos[0], "dtype", None).names
    if names is None:
        raise ValueError("Cannot read dtype field names from annotations")
    path_key = "relative_im_path" if "relative_im_path" in names else ("fname" if "fname" in names else None)
    if path_key is None:
        raise ValueError(f"No path field found. dtype.names={names}")

    parsed = []
    for a in annos:
        rel = a[path_key]
        rel = rel.item() if hasattr(rel, "item") else rel
        if isinstance(rel, bytes):
            rel = rel.decode("utf-8")

        x1 = float(a["bbox_x1"].item())
        y1 = float(a["bbox_y1"].item())
        x2 = float(a["bbox_x2"].item())
        y2 = float(a["bbox_y2"].item())
        cls1 = int(a["class"].item())

        test_flag = int(a["test"].item()) if "test" in names else 0
        parsed.append((rel, (x1, y1, x2, y2), cls1, test_flag))

    return parsed, mat.get("class_names", None)


def _read_class_names_from_cars_txt(repo_root: Path) -> List[str]:
    """
    你的仓库根目录有 cars.txt（你 ls 里看到的）。
    我们优先从它读取类别名。假设格式是：每行一个类别名。
    """
    cars_txt = repo_root / "cars.txt"
    if not cars_txt.exists():
        return []
    names = []
    for line in cars_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        # 兼容 "1 Acura Integra Type R 2001" 这种：把前面编号去掉
        parts = s.split()
        if parts and parts[0].isdigit():
            s = " ".join(parts[1:])
        names.append(s)
    return names


def _get_image_size(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def _link_or_copy(src: Path, dst: Path, mode: str):
    _ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(str(src), str(dst))
    elif mode == "copy":
        shutil.copy2(str(src), str(dst))
    else:
        raise ValueError("mode must be 'symlink' or 'copy'")


def build_coco_and_odvg(
    repo_root: Path,
    ds_root: Path,
    mode: str = "symlink",
):
    """
    ds_root: data/stanford_cars
    读取：
      - ds_root/cars_annos.mat (train)
      - ds_root/cars_test_annos_withlabels.mat (val)
      - ds_root/images/all/xxx.jpg (原图)
    输出：
      - ds_root/train_images/
      - ds_root/val_images/
      - ds_root/annotations/label_map.json
      - ds_root/annotations/train.jsonl (odvg)
      - ds_root/annotations/val.json (coco, category_id 从0开始)
    """
    ann_dir = ds_root / "annotations"
    train_img_dir = ds_root / "train_images"
    val_img_dir = ds_root / "val_images"
    all_img_dir = ds_root / "images" / "all"

    _ensure_dir(ann_dir)
    _ensure_dir(train_img_dir)
    _ensure_dir(val_img_dir)

    train_mat = ds_root / "cars_annos.mat"
    if not train_mat.exists():
        raise FileNotFoundError(f"Missing {train_mat}")

    all_items, class_names_mat = _load_mat_annotations(train_mat)

    train_items = [(p, b, c) for (p, b, c, t) in all_items if t == 0]
    val_items   = [(p, b, c) for (p, b, c, t) in all_items if t == 1]


    # 优先使用 mat 自带的 class_names
    class_names = []
    if class_names_mat is not None:
        # class_names_mat 形状常见 (1, K) 的 object array
        cn = class_names_mat
        if hasattr(cn, "ndim") and cn.ndim == 2:
            cn = cn[0]
        class_names = []
        for x in cn:
            x = x.item() if hasattr(x, "item") else x
            if isinstance(x, bytes):
                x = x.decode("utf-8")
            class_names.append(str(x).strip())

    # mat 里没有才 fallback 到 cars.txt
    if not class_names:
        class_names = _read_class_names_from_cars_txt(repo_root)

    if not class_names:
        max_cls = max([c for _, _, c in train_items + val_items])
        class_names = [f"class_{i:03d}" for i in range(1, max_cls + 1)]


    # Stanford Cars class 通常从 1..K，我们输出 label_map.json 需从 0..K-1
    K = len(class_names)
    label_map = {str(i): class_names[i] for i in range(K)}  # 0-based
    (ann_dir / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False), encoding="utf-8")

    # ---------- helper: build COCO dict ----------
    def build_coco(split_items, split_img_dir: Path):
        images = []
        annotations = []
        img_id_map: Dict[str, int] = {}
        ann_id = 1

        for fname, (x1, y1, x2, y2), cls1 in tqdm(split_items, desc=f"Prepare {split_img_dir.name}"):
            # rel_path 可能是 "car_ims/000001.jpg"
            rel_path = Path(fname)
            cand1 = all_img_dir / rel_path          # 如果 all 里保持子目录
            cand2 = all_img_dir / rel_path.name     # 如果 all 里是扁平化文件名
            src = cand1 if cand1.exists() else cand2
            if not src.exists():
                raise FileNotFoundError(f"Image not found: tried {cand1} and {cand2}")


            dst_name = Path(fname).name
            dst = split_img_dir / dst_name
            _link_or_copy(src, dst, mode=mode)

            if fname not in img_id_map:
                img_id = len(img_id_map) + 1
                img_id_map[fname] = img_id
                w, h = _get_image_size(dst)
                images.append({
                    "id": img_id,
                    "file_name": dst_name,  # 相对于 split_img_dir
                    "width": w,
                    "height": h
                })
            else:
                img_id = img_id_map[fname]

            # COCO bbox: [x, y, w, h]
            bx = max(0.0, x1)
            by = max(0.0, y1)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)

            # category_id 从0开始（教程强调这一点，避免 eval=0 的坑）
            cls0 = cls1 - 1
            if not (0 <= cls0 < K):
                raise ValueError(f"class id out of range: cls1={cls1}, K={K}")

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls0,
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

        categories = [{"id": i, "name": class_names[i]} for i in range(K)]
        return {"images": images, "annotations": annotations, "categories": categories}

    # val.json (COCO)
    coco_val = build_coco(val_items, val_img_dir)
    (ann_dir / "val.json").write_text(json.dumps(coco_val, ensure_ascii=False), encoding="utf-8")

    # train.jsonl (ODVG)
    # 这里我们直接生成 ODVG jsonl，不再依赖外部 coco2odvg 工具，格式对齐教程示例：
    # {"filename": "...", "height": H, "width": W, "detection": {"instances": [{"bbox":[x1,y1,x2,y2],"label":id,"category":"name"} ...]}}
    # 注意 bbox 用 [x1,y1,x2,y2]
    # 需要把同一张图的多个 instance 聚合到同一行
    train_group: Dict[str, Dict] = {}
    for fname, (x1, y1, x2, y2), cls1 in tqdm(train_items, desc="Build train.jsonl"):
        dst = train_img_dir / fname
        if not dst.exists():
            # 先确保图片存在（软链/复制）
            # rel_path 可能是 "car_ims/000001.jpg"
            rel_path = Path(fname)
            cand1 = all_img_dir / rel_path          # 如果 all 里保持子目录
            cand2 = all_img_dir / rel_path.name     # 如果 all 里是扁平化文件名
            src = cand1 if cand1.exists() else cand2
            if not src.exists():
                raise FileNotFoundError(f"Image not found: tried {cand1} and {cand2}")
            _link_or_copy(src, dst, mode=mode)

        if fname not in train_group:
            w, h = _get_image_size(dst)
            train_group[fname] = {
                "filename": fname,
                "height": h,
                "width": w,
                "detection": {"instances": []}
            }

        cls0 = cls1 - 1
        inst = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "label": int(cls0),
            "category": class_names[cls0]
        }
        train_group[fname]["detection"]["instances"].append(inst)

    train_jsonl_path = ann_dir / "train.jsonl"
    with train_jsonl_path.open("w", encoding="utf-8") as f:
        for fname, obj in train_group.items():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n[DONE] Generated:")
    print(f"  - {ann_dir / 'label_map.json'}")
    print(f"  - {ann_dir / 'train.jsonl'}")
    print(f"  - {ann_dir / 'val.json'}")
    print(f"  - {train_img_dir}/  ({mode})")
    print(f"  - {val_img_dir}/    ({mode})")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    ds_root = repo_root / "data" / "stanford_cars"

    # 默认用软链接，省空间。若你想真实复制，把 mode 改成 copy
    mode = os.environ.get("CARS_SPLIT_MODE", "symlink").strip().lower()
    build_coco_and_odvg(repo_root=repo_root, ds_root=ds_root, mode=mode)
