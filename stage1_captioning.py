# stage1_captioning.py
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import os
import json
from pathlib import Path
import torch
import time
import re

def run_captioning(image_folder, prompt_file, output_json):
    """
    Stage 1: Use MLLM to generate image descriptions
    """
    # 1. Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    # 2. Read prompt
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    print(f"Loaded Prompt: {prompt}")

    # 3. Get image list
    image_dir = Path(image_folder)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder '{image_folder}' does not exist.")
    
    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS])
    print(f"Found {len(image_paths)} images.")

    # 4. Inference loop
    results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {img_path}")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        if i%100==0:
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            total_inference_time += inference_time
            inference_start_time = time.time()
            # print time of every 100 images
            print(f" | Time: {inference_time:.2f}s")

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        # --- Keep your existing JSON cleaning logic ---
        caption_detail, caption_action, caption_summary, caption_all = "", "", "", ""
        try:
            json_match = re.search(r'\{[\s\S]*\}', output_text)
            if json_match:
                json_str = json_match.group(0).strip()
                if not json_str.endswith('}'):
                    last_quote = json_str.rfind('"')
                    if last_quote != -1: json_str = json_str[:last_quote+1] + '}"}'
                json_str = json_str.replace('```json', '').replace('```', '')
                parsed_json = json.loads(json_str)
                
                caption_detail = parsed_json.get("description_1", "")
                caption_action = parsed_json.get("description_2", "")
                caption_summary = parsed_json.get("description_3", "")
                caption_all = parsed_json.get("description_4", "")
        except Exception as e:
            print(f"JSON Error: {e}")
            caption_detail = output_text

        results.append({
            "image_path": img_path.name,
            "raw_output": output_text,
            "caption_detail": caption_detail,
            "caption_action": caption_action,
            "caption_summary": caption_summary,
            "caption_all": caption_all
        })

    # 5. Save results
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Stage 1 Done! Saved to {output_json}")
    
    # Release GPU memory to prevent OOM affecting subsequent steps
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # If running this file alone, it will still execute
    run_captioning("data/testflickr", "prompt.txt", "output/flickr_captions.json")
