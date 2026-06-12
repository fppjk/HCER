import json
from tqdm import tqdm
import os

def convert_coco_to_flickr_style(coco_json_path, image_prefix):
    """
    将 COCO caption JSON 转换为 Flickr30K 风格格式。
    
    参数：
    - coco_json_path: captions_train2014.json / captions_val2014.json
    - image_prefix: 图像文件前缀，如 "COCO_train2014_" 或 "COCO_val2014_"
    
    返回：
    - dataset: dict，键为图片文件名，值为 { raw:[], split:..., img_id:... }
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    # 1）建立 image_id → filename 映射
    id_to_filename = {}
    for img in images:
        file_name = f"{image_prefix}{img['id']:012d}.jpg"
        id_to_filename[img["id"]] = file_name

    # 2）建立 image_id → captions 映射
    captions = {}
    for ann in annotations:
        img_id = ann["image_id"]
        captions.setdefault(img_id, []).append(ann["caption"])

    # 3）构建输出格式
    dataset = {}
    for img in images:
        img_id = img["id"]
        # file_name = id_to_filename[img_id]
        file_name = f"{img_id:012d}.jpg"

        dataset[file_name] = {
            "raw": captions.get(img_id, []),  # 有些可能没有 caption
            "split": "train" if "train" in image_prefix else "val",
            "img_id": img_id
        }

    return dataset

if __name__ == "__main__":

    # ====== 修改为你的路径 ======
    coco_train_json = "data/coco2017/annotations/captions_train2017.json"
    coco_val_json   = "data/coco2017/annotations/captions_val2017.json"

    # ====== 生成 train 数据 ======
    # train_data = convert_coco_to_flickr_style(
    #     coco_json_path=coco_train_json,
    #     image_prefix="COCO_train2014_"
    # )
    # with open("train_coco2017.json", "w", encoding="utf-8") as f:
    #     json.dump(train_data, f, indent=4, ensure_ascii=False)

    # ====== 生成 val 数据 ======
    val_data = convert_coco_to_flickr_style(
        coco_json_path=coco_val_json,
        image_prefix="COCO_val2017_"
    )
    with open("datasets/val_coco2017.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)

    print("转换完成！输出为：")
    # print(" - coco2014_train_flickr_style.json")
    print(" - coco2017_val_flickr_style.json")
