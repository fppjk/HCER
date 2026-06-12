import json
from tqdm import tqdm
import os

def convert_coco_to_flickr_style(coco_json_path, image_prefix):
    """
    Convert COCO caption JSON to Flickr30K style format.
    
    Parameters:
    - coco_json_path: captions_train2014.json / captions_val2014.json
    - image_prefix: image file prefix, e.g., "COCO_train2014_" or "COCO_val2014_"
    
    Returns:
    - dataset: dict, keys are image filenames, values are { raw:[], split:..., img_id:... }
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    # 1) Build image_id -> filename mapping
    id_to_filename = {}
    for img in images:
        file_name = f"{image_prefix}{img['id']:012d}.jpg"
        id_to_filename[img["id"]] = file_name

    # 2) Build image_id -> captions mapping
    captions = {}
    for ann in annotations:
        img_id = ann["image_id"]
        captions.setdefault(img_id, []).append(ann["caption"])

    # 3) Build output format
    dataset = {}
    for img in images:
        img_id = img["id"]
        # file_name = id_to_filename[img_id]
        file_name = f"{img_id:012d}.jpg"

        dataset[file_name] = {
            "raw": captions.get(img_id, []),  # some may have no captions
            "split": "train" if "train" in image_prefix else "val",
            "img_id": img_id
        }

    return dataset

if __name__ == "__main__":

    # ====== Change to your paths ======
    coco_train_json = "data/coco2017/annotations/captions_train2017.json"
    coco_val_json   = "data/coco2017/annotations/captions_val2017.json"

    # ====== Generate train data ======
    # train_data = convert_coco_to_flickr_style(
    #     coco_json_path=coco_train_json,
    #     image_prefix="COCO_train2014_"
    # )
    # with open("train_coco2017.json", "w", encoding="utf-8") as f:
    #     json.dump(train_data, f, indent=4, ensure_ascii=False)

    # ====== Generate val data ======
    val_data = convert_coco_to_flickr_style(
        coco_json_path=coco_val_json,
        image_prefix="COCO_val2017_"
    )
    with open("datasets/val_coco2017.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)

    print("Conversion completed! Output:")
    # print(" - coco2014_train_flickr_style.json")
    print(" - coco2017_val_flickr_style.json")
