import os
import json
import numpy as np
import torch
import re
from tqdm import tqdm
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
# import sys
# sys.path.append("Qwen/Qwen3_VL_Reranker_8B")
# from scripts.qwen3_vl_reranker import Qwen3VLReranker
# import sys
# import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, "Qwen/Qwen3_VL_Reranker_8B/scripts"))
# sys.path.append(os.path.join(BASE_DIR, "Qwen/Qwen3_VL_Reranker_8B"))
from Qwen.Qwen3_VL_Reranker_8B.scripts.qwen3_vl_reranker import Qwen3VLReranker
from Qwen.Qwen3_VL_Reranker_2B.scripts.qwen3_vl_reranker import Qwen3VLReranker
from pathlib import Path
from PIL import Image
import time
from datetime import datetime


class QwenReranker:
    def __init__(self, output_dir, gt_path, image_root, model_path="Qwen/Qwen3_VL_Reranker_2B", top_k=10):
        self.output_dir = output_dir
        self.gt_path = gt_path
        self.model_path = model_path
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.image_root = Path(image_root) # 保存图片根路径
        # 验证GPU
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU found, using CPU. This will be very slow for large datasets!")
        # === 1. 加载粗排结果 ===
        print("Loading coarse ranking matrix and indices...")
        
        # 假设粗排结果文件路径
        sim_matrix_path = os.path.join(output_dir, "final_similarity_matrix.npy")
        indices_path = os.path.join(output_dir, "matrix_indices.json")
        
        if not os.path.exists(sim_matrix_path) or not os.path.exists(indices_path):
             raise FileNotFoundError(f"Required rough ranking files not found in {output_dir}. Please ensure 'final_similarity_matrix.npy' and 'matrix_indices.json' exist.")
             
        self.sim_matrix = np.load(sim_matrix_path)
        self.image_root = Path(image_root) # 保存图片根路径
        with open(indices_path, "r") as f:
            indices = json.load(f)
            self.query_ids = indices["row_ids_query"]    # Query 图片路径 (例如 ?????'data/testflickr/10000.jpg'??????????)
            self.target_ids = indices["col_ids_target"]  # Target 图片路径
            
        print(f"Loaded {len(self.query_ids)} queries and {len(self.target_ids)} targets.")

        # === 2. 加载 GT 文本 (用于把 Target ID 转换回文本) ===
        print("Loading GT captions...")
        self.gt_captions_map = self._load_gt_captions(gt_path)

        # === 3. 加载 Qwen3-VL 模型 ===
        print(f"Loading Qwen3-VL-Reranker model from {model_path}...")
        # self.model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     model_path, 
        #     dtype="auto", 
        #     device_map="auto"
        #     # 💡 推荐启用 Flash Attention 2 以提速和省显存
        #     # attn_implementation="flash_attention_2", 
        #     # dtype=torch.bfloat16, 
        # )
        # self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLReranker(
            model_name_or_path=model_path,
            # 推荐
            dtype=torch.bfloat16,#使用 bfloat16 数据类型可以在保持模型性能的同时显著降低显存占用，特别适合大模型如 Qwen3-VL-Reranker。
            # attn_implementation="flash_attention_2"#这个参数是Qwen3VLReranker特有的，用于启用 Flash Attention 2 来提升推理速度和降低显存占用
        )
        
        
    def _load_gt_captions(self, gt_path):
        """ 读取 GT 文件，构建 '图片ID -> 文本列表' 的映射 """
        # 假设 GT 文件格式为 { "image_path": {"raw": ["cap1", "cap2", ...]} }
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        
        caption_map = {}
        for img_path, data in gt_data.items():
            # 使用列表中的所有文本描述，这里我们只拼接前三句以保留语义多样性
            caption_map[img_path] = " ".join(data.get("raw", [])[:5]) 
            
        return caption_map

    def get_candidate_texts(self, target_img_id):
        """ 获取某个 Target 图片对应的 GT 文本描述 """
        # 注意：这里的 target_img_id 必须与 self.gt_captions_map 中的 key 格式一致
        return self.gt_captions_map.get(target_img_id, "No description available.")

    def run_reranking(self):
        """ 主循环：对每个 Query 进行重排序 """
        print(f"Starting Reranking (K={self.top_k}) on {len(self.query_ids)} queries...")
        
        correct_count_old = 0
        correct_count_new = 0 # 计算 R@1 的正确数量
        correct_count_r5 = 0 # 计算 R@5 的正确数量
        correct_count_r10 = 0 # 计算 R@10 的正确数量
        total = len(self.query_ids)
        # options = ["A", "B", "C", "D", "E"][:self.top_k]
        options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:self.top_k]
        start_time = time.time()
        
        rerank_logs = []
        
        # 遍历每一个 Query
        for i, query_img_path in tqdm(enumerate(self.query_ids), total=total):
            # --- 1. 获取 Embedding 检索的 Top-K ---
            scores = self.sim_matrix[i]
            top_k_indices = np.argsort(-scores)[:self.top_k]
            candidate_img_ids = [self.target_ids[idx] for idx in top_k_indices]# 获取 Top-K 的 Target 图片 ID
            
            # 真实 GT ID (假设 Query Image Path == Target Image Path)
            gt_truth_id = query_img_path 
            
            # 旧的 R@1
            # is_old_correct = (candidate_img_ids[0] == gt_truth_id)
            # if is_old_correct:
            #     correct_count_old += 1
                
            # --- 2. 构造  Prompt ---
            
            full_query_img_path = self.image_root / query_img_path


            # 按照Reranker模型的输入格式
            documents = []

            for cand_id in candidate_img_ids:
                text_content = self.get_candidate_texts(cand_id)
                documents.append({"text": text_content})

            inputs = {
                "instruction": "Retrieve the most semantically relevant caption describing the given image.",
                "query": {
                    "image": str(full_query_img_path)
                },
                "documents": documents,
                "fps": 1.0
            }

            # --- 3. 模型推理 ---

            
            reranker_scores = self.model.process(inputs)

            # --- 4. 解析结果 & 计算新指标 ---
            # pred_idx = -1
            # clean_output = output_text
            # pred_idx = int(np.argmax(reranker_scores))
            # selected_img_id = candidate_img_ids[pred_idx]
            


            # Reranker模型直接输出每个选项的分数，我们选择分数最高的那个作为预测结果
            sorted_indices = np.argsort(-np.array(reranker_scores))
            # R@1分数
            pred_idx = int(np.argmax(reranker_scores))
            selected_img_id = candidate_img_ids[pred_idx]
            # sorted_indices = np.argsort(-np.array(reranker_scores))

            # 是否命中 GT
            is_new_correct = (selected_img_id == gt_truth_id)
            if is_new_correct:
                correct_count_new += 1

            # R@5 分数
            top5_indices = sorted_indices[:5]
            top5_ids = [candidate_img_ids[idx] for idx in top5_indices]

            is_r5_correct = (gt_truth_id in top5_ids)
            if is_r5_correct:
                correct_count_r5 += 1

            # R@10 分数
            top10_indices = sorted_indices[:10]
            top10_ids = [candidate_img_ids[idx] for idx in top10_indices]
            is_r10_correct = (gt_truth_id in top10_ids)
            if is_r10_correct:
                correct_count_r10 += 1
            # 日志记录（注意：没有 qwen_output / option 了）
            rerank_logs.append({
                "query": query_img_path,
                "new_correct": is_new_correct,
                "r5_correct": is_r5_correct,
                "scores": [float(s) for s in reranker_scores],   # ⭐ 很重要，方便你后面做分析
                "pred_idx": pred_idx,
                "gt_truth_id": gt_truth_id,
                "selected_id": selected_img_id,
                "candidates": candidate_img_ids
            })

        # === 5. 输出最终结果 ===
        print("\n" + "="*40)
        print("🎉 Reranking Complete with Qwen3-VL-Reranker")
        print("="*40)
        print(f"Total Queries: {total}")
        print(f"Reranked R@1: {correct_count_new / total:.4f}")
        print(f"Reranked R@5: {correct_count_r5 / total:.4f}")
        print(f"Reranked R@10: {correct_count_r10 / total:.4f}")

        # 保存日志
        log_path = os.path.join(self.output_dir, f"I2T_rerank_results_k{self.top_k}.json")
        # with open(log_path, "w", encoding="utf-8") as f:
        #     json.dump(rerank_logs, f, indent=2, ensure_ascii=False)
        # print(f"Detailed logs saved to {log_path}")

        # 计算总时间
        end_time = time.time()
        total_time = end_time - start_time

        # 查询速度（queries per second）
        qps = total / total_time if total_time > 0 else 0

        # 时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 模型名称（自动从路径取）
        model_name = os.path.basename(self.model_path.strip("/"))

        # 任务类型（你可以手动写，也可以参数化）
        task_type = "I2T"  # 或 "I2T"

        r1 = correct_count_new / total
        r5 = correct_count_r5 / total
        r10 = correct_count_r10 / total
        with open(log_path, "w", encoding="utf-8") as f:
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


if __name__ == "__main__":
    # 假设粗排结果 (final_similarity_matrix.npy 和 matrix_indices.json) 放在 'output/matrix_output_i2t' 目录下
    OUTPUT_DIR = "output/coco15/matrix_output_i2t"
    # 您的 GT 标注文件路径，用于将 Target ID 映射回文本内容
    GT_PATH = "datasets/val_coco2017.json"
    # GT_PATH = "datasets/test_flickr30k.json"
    IMAGE_ROOT = "data/coco2017/val2017"
    # IMAGE_ROOT = "data/testflickr"
    MODEL_PATH = "Qwen/Qwen3_VL_Reranker_2B"
    TOP_K = 15 # 从粗排中选择前 k 个进行精排

    try:
        reranker = QwenReranker(OUTPUT_DIR, GT_PATH, IMAGE_ROOT, MODEL_PATH, TOP_K)
        reranker.run_reranking()
        # reranker.run_stability_test_i2t(n_runs=5, subset_size=100)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("请检查 OUTPUT_DIR 和 GT_PATH 配置是否正确，以及粗排结果文件是否存在。")