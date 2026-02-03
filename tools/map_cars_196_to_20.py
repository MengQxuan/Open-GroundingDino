import os, json, re
from pathlib import Path
from typing import Dict, List, Tuple

# ===== 20-class taxonomy (fixed order) =====
TARGET_LABELS = [
    "sedan",
    "coupe",
    "hatchback",
    "wagon",
    "suv",
    "pickup truck",
    "van",
    "minivan",
    "convertible",
    "sports car",
    "supercar",
    "roadster",
    "muscle car",
    "compact car",
    "luxury car",
    "off-road vehicle",
    "classic car",
    "police car",
    "race car",
    "other car",
]
LABEL_TO_ID = {n: i for i, n in enumerate(TARGET_LABELS)}

def normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_any(s: str, kws: List[str]) -> bool:
    return any(k in s for k in kws)

def map_196_name_to_20(name: str) -> str:
    """
    Heuristic mapping using keywords from Stanford Cars class names.
    You can refine rules later; default fallback is "other car".
    """
    s = normalize(name)

    # special / rare
    if "police" in s:
        return "police car"
    if "race" in s or "racing" in s:
        return "race car"

    # body style keywords in dataset are quite informative:
    # Sedan, Coupe, Hatchback, Wagon, SUV, Convertible, Van, Minivan,
    # plus pickup cues like Crew Cab/Regular Cab/Extended Cab/SuperCab/Quad Cab/SUT
    if "minivan" in s or "town and country" in s:
        return "minivan"
    if "van" in s:
        return "van"

    # pickup truck cues
    if has_any(s, ["crew cab", "regular cab", "extended cab", "supercab", "quad cab", "pickup", "1500", "2500", "3500", "srt", "sut", "silverado", "f-150", "f-450", "ram pickup"]):
        # Not all of these are pickups, but in Stanford Cars most with cab are trucks.
        # To avoid misclassifying sedans with "srt", keep cab keywords dominant:
        if has_any(s, ["crew cab", "regular cab", "extended cab", "supercab", "quad cab", "pickup", "sut"]):
            return "pickup truck"

    if "suv" in s or "range rover" in s or "grand cherokee" in s or "wrangler" in s:
        # wrangler is more off-road, but keep it under suv unless you want off-road.
        if has_any(s, ["wrangler", "offroad", "off-road"]):
            return "off-road vehicle"
        return "suv"

    if "wagon" in s:
        return "wagon"
    if "hatchback" in s:
        return "hatchback"
    if "convertible" in s or "drophead" in s or "cabriolet" in s:
        return "convertible"
    if "coupe" in s:
        return "coupe"
    if "sedan" in s:
        # luxury heuristic: S-Class, Rolls-Royce Ghost/Phantom, Maybach, Bentley Mulsanne, etc.
        if has_any(s, ["rolls-royce", "maybach", "bentley", "s-class", "phantom", "ghost", "mulsanne"]):
            return "luxury car"
        return "sedan"

    # performance / niche types
    if has_any(s, ["superleggera", "aventador", "reventon", "veyron", "mclaren", "bugatti"]):
        return "supercar"
    if has_any(s, ["ferrari", "lamborghini", "porsche", "corvette", "vantage", "r8", "mp4-12c"]):
        return "sports car"
    if "roadster" in s:
        return "roadster"
    if has_any(s, ["muscle", "challenger", "camaro", "mustang"]):
        return "muscle car"

    # classic / vintage (very rough: old year)
    m = re.search(r"(19\d{2}|20\d{2})$", s)
    if m:
        year = int(m.group(1))
        if year <= 1995:
            return "classic car"

    # compact heuristic
    if has_any(s, ["fiat 500", "smart fortwo", "c30", "golf", "beetle"]):
        return "compact car"

    return "other car"

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_label_map(out_path: Path):
    label_map = {str(i): TARGET_LABELS[i] for i in range(len(TARGET_LABELS))}
    out_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

def load_label_map_196(path: Path) -> Dict[str, str]:
    return json.load(open(path, "r", encoding="utf-8"))

def convert_train_jsonl(train_jsonl: Path, label_map_196: Dict[str, str], out_jsonl: Path):
    out_lines = []
    for line in train_jsonl.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        insts = obj["detection"]["instances"]
        for ins in insts:
            old_id = int(ins["label"])
            old_name = label_map_196[str(old_id)]
            new_name = map_196_name_to_20(old_name)
            ins["label"] = LABEL_TO_ID[new_name]
            ins["category"] = new_name
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    out_jsonl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

def convert_val_coco(val_json: Path, label_map_196: Dict[str, str], out_json: Path):
    data = json.load(open(val_json, "r", encoding="utf-8"))

    # overwrite categories to 20-class
    data["categories"] = [{"id": i, "name": TARGET_LABELS[i]} for i in range(len(TARGET_LABELS))]

    # remap each annotation category_id
    for ann in data["annotations"]:
        old_id = int(ann["category_id"])
        old_name = label_map_196[str(old_id)]
        new_name = map_196_name_to_20(old_name)
        ann["category_id"] = LABEL_TO_ID[new_name]

    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def main():
    repo = Path(__file__).resolve().parents[1]
    src = repo / "data" / "stanford_cars"
    dst = repo / "data" / "stanford_cars_20cls"

    # source paths
    src_ann = src / "annotations"
    train_jsonl = src_ann / "train.jsonl"
    val_json = src_ann / "val.json"
    label_map_196_path = src_ann / "label_map.json"

    if not train_jsonl.exists() or not val_json.exists() or not label_map_196_path.exists():
        raise FileNotFoundError("Missing one of train.jsonl / val.json / label_map.json in data/stanford_cars/annotations")

    label_map_196 = load_label_map_196(label_map_196_path)

    # destination dirs
    dst_ann = dst / "annotations"
    ensure(dst_ann)

    # write 20-class label map
    write_label_map(dst_ann / "label_map.json")

    # convert annotations
    convert_train_jsonl(train_jsonl, label_map_196, dst_ann / "train.jsonl")
    convert_val_coco(val_json, label_map_196, dst_ann / "val.json")

    # link images (reuse your existing split dirs)
    # We don't copy: we just point dataset root to original train_images/val_images to save space.
    # If you prefer separate dirs, tell me and I'll give you a linker script.
    cfg = {
        "train": [{
            "root": str(src / "train_images"),
            "anno": str(dst_ann / "train.jsonl"),
            "label_map": str(dst_ann / "label_map.json"),
            "dataset_mode": "odvg"
        }],
        "val": [{
            "root": str(src / "val_images"),
            "anno": str(dst_ann / "val.json"),
            "label_map": None,
            "dataset_mode": "coco"
        }]
    }
    out_cfg = repo / "config" / "datasets_stanfordcars_20cls_odvg.json"
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=4), encoding="utf-8")

    print("[DONE] 196->20 mapping finished.")
    print("New dataset dir:", dst)
    print("  -", dst_ann / "label_map.json")
    print("  -", dst_ann / "train.jsonl")
    print("  -", dst_ann / "val.json")
    print("New dataset cfg:", out_cfg)

if __name__ == "__main__":
    main()
