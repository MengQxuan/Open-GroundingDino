# 从COCO数据集中生成训练子集和验证子集
import argparse, json, random, os
from collections import defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)

def coco_to_xyxy(bbox):
    x, y, w, h = bbox
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]

def make_subset(inst_path, img_count, seed):
    coco = load_json(inst_path)
    random.seed(seed)

    images = coco["images"]
    anns = coco["annotations"]

    # sample images
    chosen_imgs = random.sample(images, k=min(img_count, len(images)))
    chosen_img_ids = set(img["id"] for img in chosen_imgs)

    # filter annotations for chosen images
    chosen_anns = [a for a in anns if a["image_id"] in chosen_img_ids]

    # keep categories as is (full COCO categories list)
    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": chosen_imgs,
        "annotations": chosen_anns,
        "categories": coco.get("categories", []),
    }
    return subset

def coco_instances_to_odvg_jsonl(coco_inst, out_jsonl, use_coco_id=False, label_map=None):
    """
    ODVG jsonl format expected by this repo:
    {
      "filename": "...",
      "height": H,
      "width": W,
      "detection": {"instances":[{"bbox":[x1,y1,x2,y2],"label":id,"category":name}, ...]}
    }
    """
    # map category_id -> name
    cat_id2name = {c["id"]: c["name"] for c in coco_inst["categories"]}
    # group anns by image_id
    img2anns = defaultdict(list)
    for a in coco_inst["annotations"]:
        img2anns[a["image_id"]].append(a)

    import jsonlines
    with jsonlines.open(out_jsonl, "w") as w:
        for img in coco_inst["images"]:
            ins = []
            for a in img2anns.get(img["id"], []):
                cid = a["category_id"]
                name = cat_id2name.get(cid, str(cid))
                # label: if label_map is provided (0-79), map by name
                if label_map is not None:
                    # label_map: {"0":"person", ...}
                    # find key whose value == name
                    # build reverse once outside would be faster but ok for subset
                    rev = {v: int(k) for k, v in label_map.items()}
                    label = rev[name]
                else:
                    label = cid if use_coco_id else cid
                ins.append({"bbox": coco_to_xyxy(a["bbox"]), "label": label, "category": name})

            w.write({
                "filename": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "detection": {"instances": ins}
            })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_inst", required=True)
    ap.add_argument("--val_inst", required=True)
    ap.add_argument("--train_n", type=int, default=5000)
    ap.add_argument("--val_n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_map", default="config/coco2017_label_map.json")
    args = ap.parse_args()

    label_map = load_json(args.label_map)

    # train subset (instances json)
    train_subset = make_subset(args.train_inst, args.train_n, args.seed)
    out_train_inst = os.path.join(args.out_dir, f"instances_train2017_subset_{args.train_n}.json")
    save_json(train_subset, out_train_inst)

    # val subset (instances json)
    val_subset = make_subset(args.val_inst, args.val_n, args.seed)
    out_val_inst = os.path.join(args.out_dir, f"instances_val2017_subset_{args.val_n}.json")
    save_json(val_subset, out_val_inst)

    # train subset -> odvg jsonl (use label_map to produce 0-79 labels)
    out_train_odvg = os.path.join(args.out_dir, f"odvg_train2017_subset_{args.train_n}.jsonl")
    coco_instances_to_odvg_jsonl(train_subset, out_train_odvg, label_map=label_map)

    print("Wrote:")
    print(" ", out_train_inst)
    print(" ", out_val_inst)
    print(" ", out_train_odvg)

if __name__ == "__main__":
    main()
