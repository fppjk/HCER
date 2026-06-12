import os
import json
import numpy as np
import torch
from tqdm import tqdm
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from Qwen.Qwen3_VL_Reranker_8B.scripts.qwen3_vl_reranker import Qwen3VLReranker
from Qwen.Qwen3_VL_Reranker_2B.scripts.qwen3_vl_reranker import Qwen3VLReranker
from pathlib import Path
import time
from datetime import datetime

class QwenRerankerT2I:
    def __init__(self, output_dir, gt_path, image_root, model_path="Qwen/Qwen3_VL_Reranker_8B", top_k=15):
        self.output_dir = output_dir
        self.gt_path = gt_path
        self.model_path = model_path
        self.top_k = top_k
        self.image_root = Path(image_root)
        
        # === 1. 加载 T2I 粗排结果 ===
        print("Loading T2I coarse ranking matrix and indices...")
        sim_matrix_path = os.path.join(output_dir, "final_similarity_matrix.npy")
        indices_path = os.path.join(output_dir, "matrix_indices.json")
        
        if not os.path.exists(sim_matrix_path) or not os.path.exists(indices_path):
            raise FileNotFoundError(f"Required T2I files not found in {output_dir}.")
            
        self.sim_matrix = np.load(sim_matrix_path) # (5000, 1000)
        
        with open(indices_path, "r") as f:
            indices = json.load(f)
            self.query_img_ids = indices["row_ids_query"] # 5000条Query对应的GT图像ID
            self.target_img_ids = indices["col_ids_target"] # 1000张候选图像ID
            
        # === 2. 加载 GT 文本 ===
        self.gt_sentences = self._load_gt_sentences(gt_path)

        # === 3. 加载模型 ===
        print(f"Loading Qwen3-VL-Reranker model from {model_path}...")
        # self.model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     model_path, dtype="auto", device_map="auto"
        # )
        # self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLReranker(
            model_name_or_path=model_path,
            dtype=torch.bfloat16,#使用 bfloat16 数据类型可以在保持模型性能的同时显著降低显存占用，特别适合大模型如 Qwen3-VL-Reranker。
            # attn_implementation="flash_attention_2"#这个参数是Qwen3VLReranker特有的，用于启用 Flash Attention 2 来提升推理速度和降低显存占用
        )
        
    def _load_gt_sentences(self, gt_path):
        """ 按顺序获取 5000 个 Query 的原始文本 """
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        
        img_to_caps = {img: data.get("raw", []) for img, data in gt_data.items()}
        query_texts = []
        counters = {}
        for img_id in self.query_img_ids:
            counters[img_id] = counters.get(img_id, 0)
            caps = img_to_caps.get(img_id, ["No caption"])
            query_texts.append(caps[counters[img_id] % len(caps)])
            counters[img_id] += 1
        return query_texts

    def run_reranking(self):
        start_time = time.time()
        total_queries = len(self.query_img_ids)
        correct_count = 0 # R@1
        correct_count_r5 = 0 # R@5
        correct_count_r10 = 0 # R@10

        rerank_logs = []
        
        print(f"Starting T2I Reranking on {total_queries} queries...")

        for i in tqdm(range(total_queries), total=total_queries):
            
            # --- 1. Top-K ---
            sim_scores = self.sim_matrix[i]
            top_k_indices = np.argsort(-sim_scores)[:self.top_k]
            candidate_img_ids = [self.target_img_ids[idx] for idx in top_k_indices]
            
            gt_truth_id = self.query_img_ids[i]
            query_text = self.gt_sentences[i]

            # --- 2. 构造 documents（候选图像）---
            documents = []
            for cand_id in candidate_img_ids:
                img_path = self.image_root / cand_id
                
                documents.append({
                    "image": str(img_path)
                    # ⭐ 推荐加这个（效果更好）
                    # "text": self.gt_sentences[i]
                })

            # --- 3. 构造 reranker 输入 ---
            inputs = {
                "instruction": "Retrieve the most relevant image that matches the given text description.",
                "query": {
                    "text": query_text
                },
                "documents": documents,
                "fps": 1.0
            }

            # --- 4. 推理 ---
            reranker_scores = self.model.process(inputs)

            # --- 5. 获取推理结果 ---
            sorted_indices = np.argsort(-np.array(reranker_scores))
            pred_idx = int(np.argmax(reranker_scores))
            selected_id = candidate_img_ids[pred_idx]

            # --- 6. 评估 ---
            #R@1
            is_correct = (selected_id == gt_truth_id)
            if is_correct:
                correct_count += 1
            # R@5
            top5_indices = sorted_indices[:5]
            top5_ids = [candidate_img_ids[idx] for idx in top5_indices]

            is_r5_correct = (gt_truth_id in top5_ids)
            if is_r5_correct:
                correct_count_r5 += 1

            #R@10
            top10_indices = sorted_indices[:10]
            top10_ids = [candidate_img_ids[idx] for idx in top10_indices]
            is_r10_correct = (gt_truth_id in top10_ids)
            if is_r10_correct:
                correct_count_r10 += 1
            # --- 7. 日志 ---
            rerank_logs.append({
                "sentence_idx": i,
                "query": query_text,
                "is_correct": is_correct,
                "r5_correct": is_r5_correct,
                "r10_correct": is_r10_correct,
                "scores": [float(s) for s in reranker_scores],
                "pred_idx": pred_idx,
                "gt_id": gt_truth_id,
                "selected_id": selected_id
            })

        # --- 8. 结果 ---
        final_r1 = correct_count / total_queries
        final_r5 = correct_count_r5 / total_queries
        final_r10 = correct_count_r10 / total_queries
        print("\n" + "="*40)
        print(f"✅ T2I Reranking Summary (K={self.top_k})")
        print(f"Total Queries: {total_queries}")
        print(f"Correct Predictions: {correct_count}")
        print(f"Reranked T2I R@1: {final_r1:.4f}")
        print(f"Reranked T2I R@5: {final_r5:.4f}")
        print(f"Reranked T2I R@10: {final_r10:.4f}")
        print("="*40)

        save_path = os.path.join(self.output_dir, f"t2i_rerank_final_k{self.top_k}.json")
        # 计算总时间
        end_time = time.time()
        total_time = end_time - start_time

        # 查询速度（queries per second）
        qps = total_queries / total_time if total_time > 0 else 0

        # 时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 模型名称（自动从路径取）
        model_name = os.path.basename(self.model_path.strip("/"))

        # 任务类型（你可以手动写，也可以参数化）
        task_type = "T2I"  # 或 "I2T"

        r1 = final_r1
        r5 = final_r5
        r10 = final_r10
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(rerank_logs, f, indent=2, ensure_ascii=False)
        
                # 日志内容（一行）
        log_line = (
            f"[{current_time}] | Model: {model_name} | Task: {task_type} | "
            f"R@1: {r1:.4f} | R@5: {r5:.4f} | R@10: {r10:.4f} | QPS: {qps:.2f}\n"
        )

        # 写入文件
        log_file = os.path.join("", "experiment_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

        try:
            del evaluator
        except:
            pass
        # gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 🚨 请根据您的实际路径修改以下配置 🚨
    # 必须是 T2I 粗排结果 (final_similarity_matrix.npy: 5000x1000, matrix_indices.json) 所在目录
    OUTPUT_DIR = "output/flickr/matrix_output_t2i" 
    # OUTPUT_DIR = "output/coco/matrix_output_t2i"
    # 您的 GT 标注文件路径，用于获取 5000 句查询文本
    GT_PATH = "datasets/test_flickr30k.json" 
    # GT_PATH = "datasets/val_coco2017.json"
    # 图片根目录，用于加载 Top-K 候选图片
    IMAGE_ROOT = "data/testflickr"
    # IMAGE_ROOT = "data/coco2017/val2017"
    MODEL_PATH = "Qwen/Qwen3_VL_Reranker_2B" 
    TOP_K = 15 # 从粗排中选择前15个进行精排

    try:
        reranker = QwenRerankerT2I(OUTPUT_DIR, GT_PATH, IMAGE_ROOT, MODEL_PATH, TOP_K)
        reranker.run_reranking()
        # reranker.run_stability_test_t2i(n_runs=5, subset_size=100)
        # try:
        #     del reranker
        #     reranker =QwenRerankerT2I("output/alpha3/matrix_output_t2i", GT_PATH, IMAGE_ROOT, MODEL_PATH, TOP_K)
        #     # del reranker 
        #     try:
        #         del reranker
        #         reranker =QwenRerankerT2I("output/alpha7/matrix_output_t2i", GT_PATH, IMAGE_ROOT, MODEL_PATH, TOP_K)
        #     except:
        #         pass
        # except:
        #     pass
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("请检查 OUTPUT_DIR 和 GT_PATH 配置是否正确，以及 T2I 粗排结果文件是否存在。")