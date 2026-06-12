import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm
import csv

class CaptionRetrievalEvaluatorFast:
    def __init__(self, gt_path, gen_path, model_name="Model/bge-large-en-v1.5"):

        # === Load Data === 
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        with open(gen_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        # 原标签 (5 captions)
        self.gt_captions = {img: info["raw"] for img, info in gt_data.items()}
        # self.gt_captions = {
        #     img: info["raw"][:3]  # 只取前3个caption
        #     for img, info in gt_data.items()
        # }

        self.gtpj_captions = {
            img: [" ".join(info["raw"])]  
            for img, info in gt_data.items()
        }

        # 生成标签 (K captions)
        # self.gen_captions = {
        #     item["image_path"]: [
        #         item["Sentence1"],
        #         item["Sentence2"],
        #         item["Sentence3"],
        #         item["Sentence4"],
        #         item["Sentence5"]
        #     ]
        #     for item in gen_data
        # }

        # self.genpj_captions = {
        #     item["image_path"]: [" ".join([
        #         item["Sentence1"],
        #         item["Sentence2"],
        #         item["Sentence3"],
        #         item["Sentence4"],
        #         item["Sentence5"]
        #     ])]  # 关键：放入一个元素的列表
        #     for item in gen_data
        # }
        self.gen_captions = {
            item["image_path"]: [
                # item["raw_output"]
                item["caption_detail"],
                item["caption_action"],
                item["caption_summary"],
                item["caption_all"]
            ]
            for item in gen_data
        }

        self.genpj_captions = {
            item["image_path"]: [" ".join([
                # item["raw_output"]
                item["caption_detail"],
                item["caption_action"],
                item["caption_summary"],
                item["caption_all"]
            ])]  # 关键：放入一个元素的列表
            for item in gen_data
        }

        # === Load model ===
        self.model = SentenceTransformer(model_name, local_files_only=True)

        # Encode all captions once!
        print("Batch encoding captions...")

        self.gt_sent_list, self.gt_index_map = self._flatten(self.gt_captions)
        self.gen_sent_list, self.gen_index_map = self._flatten(self.gen_captions)

        # === Batch encode ===
        self.gt_emb = self.model.encode(
            self.gt_sent_list,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.gen_emb = self.model.encode(
            self.gen_sent_list,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Restore per-image embedding list
        self.gt_emb_dict = self._group_embeddings(self.gt_emb, self.gt_index_map)
        self.gen_emb_dict = self._group_embeddings(self.gen_emb, self.gen_index_map)

        # 直接拼接向量 (不进行池化)
        # self.gt_matrix, self.gt_img_list = self._concat_embeddings(self.gt_emb_dict)
        # self.gen_matrix, self.gen_img_list = self._concat_embeddings(self.gen_emb_dict)


        # Convert dict to fixed matrices for fast retrieval,就是多句话的向量进行mean pooling
        self.gt_matrix, self.gt_img_list = self._merge_per_image(self.gt_emb_dict)
        self.gen_matrix, self.gen_img_list = self._merge_per_image(self.gen_emb_dict)

        # 内部变量
        self.results = {}
        self.retrieval_results = {}

        # 处理 gtpj 和 genpj（单句） 
        # 1. 扁平化句子列表（每个图像1句）
        self.gtpj_sent_list, self.gtpj_index_map = self._flatten(self.gtpj_captions)
        self.genpj_sent_list, self.genpj_index_map = self._flatten(self.genpj_captions)

        # 2. 直接编码（单句不用池化！）
        print("Encoding single-sentence captions (gtpj/genpj)...")
        self.gtpj_emb = self.model.encode(
            self.gtpj_sent_list,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.genpj_emb = self.model.encode(
            self.genpj_sent_list,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 3. 生成矩阵（单句直接取第一个向量）
        self.gtpj_emb_dict = self._group_embeddings(self.gtpj_emb, self.gtpj_index_map)
        self.genpj_emb_dict = self._group_embeddings(self.genpj_emb, self.genpj_index_map)
        
        # 关键：用 self.gt_img_list/self.gen_img_list 保证顺序一致
        self.gtpj_matrix = np.array([self.gtpj_emb_dict[img][0] for img in self.gt_img_list])  # [num_img, D]
        self.genpj_matrix = np.array([self.genpj_emb_dict[img][0] for img in self.gen_img_list])  # [num_img, D]

        self.last_sim_matrix = None
        self.last_query_list = None
        self.last_target_list = None



    def _flatten(self, cap_dict):
        sent_list = []
        index_map = []
        for img, caps in cap_dict.items():
            for c in caps:
                sent_list.append(c)
                index_map.append(img)
        return sent_list, index_map

    def _group_embeddings(self, emb_array, index_map):
        emb_dict = {}
        for img, emb in zip(index_map, emb_array):
            emb_dict.setdefault(img, []).append(emb)
        return emb_dict

    def _merge_per_image(self, emb_dict):
        """
        对每张图像，把 (N×D) 的句子 embedding 聚合成一个图像向量
        聚合方法：Mean Pooling
        """
        img_list = list(emb_dict.keys())
        all_vec = []

        for img in img_list:
            arr = np.stack(emb_dict[img], axis=0)  # (num_sent, D)
            pooled = arr.mean(axis=0)  # Mean pooling 聚合成一个向量
            all_vec.append(pooled)

        return np.stack(all_vec, axis=0), img_list  # (num_img, D)
    
    def _merge_per_image_cumulative(self, emb_dict):
        """
        对每张图像的 N 个句子向量 (N×D)，计算 N 个累积 Mean Pooling 向量
        最后对这 N 个累积向量进行 Mean Pooling 聚合成一个图像向量 (二次池化)。
        用于 I2T 检索的 Target (1000, D)
        """
        img_list = list(emb_dict.keys())
        all_vec = []

        for img in img_list:
            emb_list = emb_dict[img] 
            
            # 1. 计算 N 个累积向量
            cumulative_vectors = []
            current_sum = np.zeros_like(emb_list[0]) # 初始化累积和
            
            for i, emb in enumerate(emb_list):
                current_sum += emb # 累加向量
                cumulative_mean = current_sum / (i + 1) # 计算平均值 (Mean Pooling)
                cumulative_vectors.append(cumulative_mean)

            # 2. 对这 N 个累积向量进行最终的 Mean Pooling (二次池化)
            arr = np.stack(cumulative_vectors, axis=0)  # (num_cumulative, D)
            final_pooled = arr.mean(axis=0)  # 最终 Mean Pooling

            all_vec.append(final_pooled)

        return np.stack(all_vec, axis=0), img_list  # (num_img, D)

    def _collect_cumulative_embeddings(self, emb_dict):
        """
        新增方法：收集所有图像的 N 个累积向量，不进行最终池化。
        用于 T2I 检索的 Query (5000, D)
        """
        all_cumulative_vec = []
        index_map_5000 = []

        for img in emb_dict.keys():
            emb_list = emb_dict[img] 
            
            # 计算 N 个累积向量
            current_sum = np.zeros_like(emb_list[0])
            
            for i, emb in enumerate(emb_list):
                current_sum += emb
                cumulative_mean = current_sum / (i + 1)
                all_cumulative_vec.append(cumulative_mean)
                index_map_5000.append(img) # 记录该向量对应的图像ID
        
        return np.stack(all_cumulative_vec, axis=0), index_map_5000 # (num_img * 5, D)

    def _concat_embeddings(self, emb_dict):
        """
        直接拼接每个图像的多个向量 (不进行池化)
        """
        img_list = list(emb_dict.keys())
        all_vec = []

        for img in img_list:
            # 拼接这个图像的所有向量
            concatenated = np.concatenate(emb_dict[img])
            all_vec.append(concatenated)
        
        return np.stack(all_vec, axis=0), img_list

    def _collect_top_k_results(self, sim_matrix, query_img_list, target_img_list, k):
        """
        收集每个 query 的 Top-K 检索结果及其分数，并记录是否成功召回（Top-K 内命中）。
        """
        results = {}
        ranks = np.argsort(-sim_matrix, axis=1)

        for i, query_img in enumerate(query_img_list):
            top_k_indices = ranks[i, :k]
            
            top_k_data = []
            is_correct = False
            for rank_idx in top_k_indices:
                top_img = target_img_list[rank_idx]
                top_score = sim_matrix[i, rank_idx]
                top_k_data.append((top_img, top_score))
                
                if top_img == query_img:
                    is_correct = True
            
            top1_score = top_k_data[0][1] if top_k_data else 0.0

            results[query_img] = {
                'top_k': top_k_data, # 存储 [(img_id, score), ...]
                'is_correct': is_correct,
                'top1_score': float(top1_score)
            }

        return results

    def _collect_top_k_results_t2i(self, sim_matrix, query_img_list_5000, target_img_list, k):
        """
        收集 T2I (5000x1000) 检索结果，并按图像ID聚合。
        对于每张图像 (Image ID)，我们取其 5 个 Query 句子中得分最高的一次检索结果。
        """
        ranks = np.argsort(-sim_matrix, axis=1) # (5000, 1000)
        
        # 存储每张 GT 图像的最佳 Top-K 结果
        best_results_per_image = {} # {img_id: {'top_k': [(img_id, score), ...], 'top1_score': score, 'is_correct': bool}}

        for i in range(sim_matrix.shape[0]): # 遍历 5000 个 Query 句子
            query_gt_img_id = query_img_list_5000[i]
            
            # 获取当前句子查询的 Top-K 结果
            top_k_indices = ranks[i, :k]
            top_k_data = []
            is_correct = False

            for rank_idx in top_k_indices:
                top_img = target_img_list[rank_idx]
                top_score = sim_matrix[i, rank_idx]
                top_k_data.append((top_img, top_score))
                
                # GT 和 Gen 图像 ID 匹配
                if top_img == query_gt_img_id:
                    is_correct = True
            
            top1_score = top_k_data[0][1] if top_k_data else 0.0

            # 聚合逻辑：如果当前结果比该图像记录的最佳结果更好 (基于 Top-1 Score)，则更新
            # 注意：T2I 的 R@K 统计方式是取 5 句中最好的结果。
            if query_gt_img_id not in best_results_per_image or top1_score > best_results_per_image[query_gt_img_id]['top1_score']:
                
                # 重新计算 R@K 的 is_correct 状态：取 5 句中只要有一句能召回，则该图像召回成功
                # 注意：这里我们只更新 Top-1 对应的 Top-K 列表
                
                # 如果当前句子召回成功 (is_correct=True) 
                # 且该图像之前未被召回，则更新为成功
                
                # 简化逻辑：我们只更新 Top-1 Score 最高的结果，并以该结果的 Top-K 列表作为代表
                
                # 在 _compute_recall_5000_t2i 中，我们已经根据 5 句话最好的结果计算了 R@K
                # 这里的目的只是为了导出 Top-1 Score 和 Top-K 列表
                
                current_is_correct = (top_k_data[0][0] == query_gt_img_id) # 当前 Top-1 是否正确
                
                best_results_per_image[query_gt_img_id] = {
                    'top_k': top_k_data, # 记录 Top-1 Score 对应的 Top-K 列表
                    'top1_score': float(top1_score),
                    'is_correct_top1': current_is_correct # 记录当前 Top-1 是否正确
                    # 注意：'is_correct' 在 R@K 统计中是 5 句中最优
                }
                
        # 最终 R@K 状态需要从 _compute_recall_5000_t2i 中获取，但为了避免重复计算，
        # 我们使用一个近似值：如果 Top-1 Score 最高的结果是正确的，则认为 Top-1 正确
        # 由于我们只做导出，这里主要关注 Top-1 结果
        return best_results_per_image


    def _avg_of_max_similarity_i2t(self):
        """
        I2T: Query=Gen (4), Target=GT (5)
        Sim(gen, gt) = Avg_i ( max_j ( Sim(gen_i, gt_j) ) )
        """
        num_imgs = len(self.gen_img_list)
        sim_matrix = np.zeros((num_imgs, num_imgs), dtype=np.float32)

        print("Calculating I2T Avg-of-Max Similarity Matrix (1000x1000)...")
        
        # 遍历 Query 图像 (Gen)
        for i, gen_img_id in enumerate(self.gen_img_list):
            gen_embs = np.stack(self.gen_emb_dict[gen_img_id], axis=0) # (4, D)
            
            # 遍历 Target 图像 (GT)
            for j, gt_img_id in enumerate(self.gt_img_list):
                gt_embs = np.stack(self.gt_emb_dict[gt_img_id], axis=0) # (5, D)
                
                # 1. 计算句子对相似度 (4, D) @ (D, 5) = (4, 5)
                pair_sim_matrix = np.matmul(gen_embs, gt_embs.T) 

                # 2. 对每个 Gen 句子 (行) 取与 GT 的最大相似度 (max_j)
                # 得到 (4,) 向量，即 [max(gen1), max(gen2), max(gen3), max(gen4)]
                max_sim_per_gen_sentence = np.max(pair_sim_matrix, axis=1) 

                # 3. 对这 4 个最大值取平均 (Avg_i)
                final_sim = np.mean(max_sim_per_gen_sentence)
                sim_matrix[i, j] = final_sim
        
        return sim_matrix
    

    # I2T (对生成文本向量与 GT 文本向量进行检索)
    def evaluate_i2t(self):
        print("Running I2T (matrix retrieval)...")

        # === 1. 特征融合 (Feature Fusion) ===
        # Gen 侧 (Query) 使用 ( Mean Pooling) + (单句拼接向量) 的平均
        combined_gen_matrix = self.gen_matrix*0.9 + self.genpj_matrix*0.1
        # combined_gen_matrix = self.gen_matrix

        # 归一化 Gen Query
        combined_gen_matrix = normalize(combined_gen_matrix, axis=1, norm='l2')

        # === 2. GT 侧 (Target) 使用所有 5000 条独立句子，不做 pooling ===
        # self.gt_emb 已经在 __init__ 中做了 L2 归一化 (normalize_embeddings=True)
        gt_target_emb = self.gt_emb  # (5000, D)

        # 计算相似度矩阵: (1000, D) @ (D, 5000) = (1000, 5000)
        sim_matrix = np.matmul(combined_gen_matrix, gt_target_emb.T)
        print(f"I2T 相似度矩阵大小: {sim_matrix.shape}")

        # 统计结果 ===
        # 注意：target_img_list 使用 gt_index_map (5000个句子各自对应的图像ID)
        self.retrieval_results = self._collect_top_k_results(
            sim_matrix,
            self.gen_img_list,      # Query: 1000 个 Gen 图像
            self.gt_index_map,      # Target: 5000 条 GT 句子 (每条句子对应其图像ID)
            k=5
        )
        self.last_sim_matrix = sim_matrix
        self.last_query_list = self.gen_img_list      # 1000
        self.last_target_list = self.gt_index_map     # 5000
        return self._compute_recall(sim_matrix, self.gen_img_list, self.gt_index_map)
                
    # T2I (text query → image)
    def evaluate_t2i(self):
        print("Running T2I (matrix retrieval) using 5000 cumulative vectors as query...")

        # 1. Gen 侧 (Target) 使用 (普通 Mean Pooling) + (单句拼接向量) 的平均
        # combined_gen_matrix = (self.gen_matrix + self.genpj_matrix) / 2.0
        combined_gen_matrix = self.gen_matrix*0.5 + self.genpj_matrix*0.5
        # combined_gen_matrix = self.gen_matrix
        combined_gen_matrix = normalize(combined_gen_matrix, axis=1, norm='l2')
        
        # 2. GT 侧 (Query) 使用 5000 个累积句子向量
        # self.gt_cumulative_emb 已经在 __init__ 中生成并归一化
        # gt_emb_normalized = self.gt_cumulative_emb
        gt_emb_normalized = normalize(self.gt_emb, axis=1, norm='l2')
        
        # 3. 计算相似度矩阵: (5000, D) @ (D, 1000) = (5000, 1000)
        sim_matrix_5000 = np.matmul(gt_emb_normalized, combined_gen_matrix.T) 
        print(f"T2I 相似度矩阵大小: {sim_matrix_5000.shape}")

        # 4. 收集 Top-K 结果，用于统计和导出
        self.retrieval_results_t2i = self._collect_top_k_results_t2i(
            sim_matrix_5000,
            self.gt_index_map, # Query 列表 (5000个累积句子对应的图像路径)
            self.gen_img_list, # Target 列表 (1000个 Gen 图像路径)
            k=5
        )

        # 5. 更新 Last 变量，以便保存矩阵
        self.last_sim_matrix = sim_matrix_5000
        self.last_query_list = self.gt_index_map # T2I 的 Query 是 5000 个句子
        self.last_target_list = self.gen_img_list # T2I 的 Target 是 1000 个图像
        
        # 4. 计算 Recall
        return self._compute_recall_5000_t2i(
            sim_matrix_5000, 
            self.gt_index_map, # Query 列表 (5000个累积句子对应的图像路径)
            self.gen_img_list              # Target 列表 (1000个 Gen 图像路径)
        )
    
    def evaluate_max_sim_t2i_5000(self):
        """
        执行 T2I 检索，使用 5000 个 GT 句子作为 Query，1000 个 Gen 组作为 Target。
        相似度计算：Sim(gt_j, Gen_img) = max(Sim(gt_j, Gen_1), ..., Sim(gt_j, Gen_4))
        最终矩阵形状：(5000, 1000)
        """
        print("Running T2I with (5000 sentences -> 1000 images) Max Similarity...")

        # --- Step 1: 准备 Query 和 Target 矩阵 ---
        # Query 矩阵 (GT 所有句子): (5000, D)
        query_emb = self.gt_emb 

        # Target 矩阵 (Gen 所有句子): (4000, D)
        target_emb = self.gen_emb

        # --- Step 2: 计算所有句子对的相似度 ---
        # sim_all_pairs: (5000, D) @ (D, 4000) = (5000, 4000)
        # sim_all_pairs[j, k] 是 GT 句子 j 和 Gen 句子 k 的相似度
        sim_all_pairs = np.matmul(query_emb, target_emb.T)
        print(f"All-pairs similarity matrix size: {sim_all_pairs.shape}")
        
        # --- Step 3: 对 Gen 组进行 Max Pooling ---
        # Gen 句子列表 (self.gen_sent_list) 是按图像顺序排列的
        # [Img1_s1, Img1_s2, Img1_s3, Img1_s4, Img2_s1, ..., Img1000_s4]
        # 每 4 个 Gen 句子对应一张图像。
        
        # 将 (5000, 4000) 矩阵重塑为 (5000, 1000, 4)
        # sim_grouped[j, i, k] 是 GT 句子 j 和 Gen 图像 i 的第 k 个 Gen 句子的相似度
        sim_grouped = sim_all_pairs.reshape(sim_all_pairs.shape[0], -1, 4) # (5000, 1000, 4)

        # 对最后一维（4个Gen句子）取最大值
        # sim_matrix_t2i: (5000, 1000)
        # sim_matrix_t2i[j, i] 是 GT 句子 j 和 Gen 图像 i 的最终相似度得分 (Max Sim)
        sim_matrix_t2i = np.max(sim_grouped, axis=2) 
        print(f"Final T2I similarity matrix size: {sim_matrix_t2i.shape}")

        # --- Step 4: 计算 Recall ---
        # Query ID list 是 5000 个 GT 句子各自对应的图像 ID
        # self.gt_index_map 就是 5000 个句子对应的图像路径列表
        return self._compute_recall_5000_t2i(
            sim_matrix_t2i, 
            self.gt_index_map,   # 5000 个查询句子对应的 GT 图像 ID
            self.gen_img_list    # 1000 个 Target Gen 图像 ID
        )

    def _compute_recall_5000_t2i(self, sim_matrix, query_img_list_5000, target_img_list,return_status=False):
        # sim_matrix: (5000, 1000)
        # query_img_list_5000: 5000 个 GT 句子对应的图像路径 (Query Image ID)
        # target_img_list: 1000 个 Gen 图像路径 (Target Image ID)
        
        # 1. 对相似度矩阵按行降序排序，得到排名索引 (5000x1000)
        ranks = np.argsort(-sim_matrix, axis=1) 

        def recall_at_k_image_level(k):
            # 记录 1000 张 GT 图像中，哪些被成功召回
            # 初始化为所有图像都未被召回
            # self.gt_img_list 应该是 1000 个 GT 图像的 ID 列表
            image_recalled_status = {img_id: False for img_id in self.gt_img_list} 
            count=0
            # 遍历 5000 个句子 (矩阵的每一行)
            for i in range(sim_matrix.shape[0]):
                query_gt_img_id = query_img_list_5000[i] # 当前句子对应的 GT 图像 ID (如: img_a.jpg)
                

                # 关键优化点：如果该 GT 图像已经被它前面的句子召回了，就跳过
                # 因为我们只需要知道它是否能被召回（5个句子取最好结果），不需要重复计算
                # if image_recalled_status[query_gt_img_id]:
                #     continue
                
                # 找到当前句子查询的 Top-K 索引
                top_k_indices = ranks[i, :k]
                # 找到 Top-K 对应的 Gen 图像 ID
                top_k_gen_imgs = [target_img_list[j] for j in top_k_indices]
                
                # GT 和 Gen 图像 ID 必须匹配才能视为成功召回
                # 假设 GT 图像 ID 'img_a.jpg' 的目标是 Gen 图像 ID 'img_a.jpg' 
                # (需要确保 Gen 图像列表中存在对应的 ID)
                target_gen_img_id = query_gt_img_id 

                # 判断：如果目标 Gen 图像ID在 Top-K 检索结果中
                if target_gen_img_id in top_k_gen_imgs:
                    # 图像召回成功，设置状态
                    image_recalled_status[query_gt_img_id] = True
                    count+=1
            
            # 最终结果：被召回的图像数量 / 总图像数量 (1000)
            recalled_count = sum(image_recalled_status.values())
            # print("image recalled count:", recalled_count)
            return count/5000.0 ,image_recalled_status
        
        # 计算 R@1, R@5, R@10
        # r1 = recall_at_k_image_level(1)
        # r5 = recall_at_k_image_level(5)
        # r10 = recall_at_k_image_level(10)
        r1, status1 = recall_at_k_image_level(1)
        r5, status5 = recall_at_k_image_level(5)
        r10, status10 = recall_at_k_image_level(10)

        if return_status:
            # 返回 R@5 的状态字典作为代表，用于导出
            return r1, r5, r10, status5 
        else:
            return r1, r5, r10


    # Recall@K
    def _compute_recall(self, sim_matrix, query_img_list, target_img_list):

        ranks = np.argsort(-sim_matrix, axis=1)  # 每行由大到小排序

        def recall_at_k(k):
            correct = 0
            for i, query_img in enumerate(query_img_list):
                top_k = ranks[i, :k]
                top_k_imgs = [target_img_list[j] for j in top_k]
                if query_img in top_k_imgs:
                    correct += 1
            print(len(query_img_list))
            return correct / len(query_img_list)

        r1 = recall_at_k(1)
        r5 = recall_at_k(5)
        r10 = recall_at_k(10)

        return r1, r5, r10


    def export_correct(self, save_path, k=5):
        """
        导出 Top-K 内成功召回的样本（Query ID, Top1 检索结果, Top1 Score）。
        前提：evaluate_avg_sim_i2t(k) 必须先运行。
        """
        if not self.retrieval_results:
            print("[ERROR] 请先运行 evaluate_avg_sim_i2t() 以生成检索结果。")
            return

        rows = []
        
        for query_img, res_data in self.retrieval_results.items():
            if res_data['is_correct']: # 成功召回（在 Top-K 内）
                top1_img, top1_score = res_data['top_k'][0] # 只记录 Top-1 结果
                rows.append([query_img, top1_img, float(top1_score)])

        # 保存到 CSV
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_img", "top1_retrieved_img", "top1_score", "is_correct_in_top_k"])
            writer.writerows(rows)

        print(f"[INFO] 正确召回案例 (Top-{k} 内) 已导出到: {save_path}")

    def export_errors(self, save_path, k_sentences=5, k_images=5):
        """
        导出 Top-K_images 内未能成功召回的样本，并给出 Top-K_sentences 个最相似的 GT 句子。
        """
        if self.last_sim_matrix is None or not self.retrieval_results:
            print("[ERROR] 请先运行 evaluate_avg_sim_i2t() 以生成相似度矩阵和检索结果。")
            return

        rows = []
        
        # 1. 遍历检索结果，找到错误召回的 Query
        for query_img, res_data in self.retrieval_results.items():
            if res_data['is_correct']: # 仅处理错误召回的样本
                continue 

            # 2. 获取该 Query 在矩阵中的索引 i
            i = self.last_query_list.index(query_img)
            
            # 3. 获取 Top-k_images 的目标图像索引
            ranks_images = np.argsort(-self.last_sim_matrix[i, :]) # 图像级别的相似度排序
            top_k_image_indices = ranks_images[:k_images]
            
            # 4. 找到 Top-1 图像
            top1_image_idx = top_k_image_indices[0]
            top1_retrieved_img = self.last_target_list[top1_image_idx]
            top1_score = self.last_sim_matrix[i, top1_image_idx]
            
            # 5. 获取 Query 图像（Gen）的 embedding 矩阵
            gen_embs = np.stack(self.gen_emb_dict[query_img], axis=0) # (N_gen, D) - N_gen=4

            # 6. 获取 Top-1 错误图像（GT）的 embedding 矩阵
            gt_embs_error = np.stack(self.gt_emb_dict[top1_retrieved_img], axis=0) # (N_gt, D) - N_gt=5
            gt_captions_error = self.gt_captions[top1_retrieved_img] # 对应的 5 句 GT 文本

            # 7. 计算 Query Gen 句子和 Top-1 错误 GT 句子的所有相似度
            # pair_sim_matrix: (N_gen, N_gt) = (4, 5)
            pair_sim_matrix = np.matmul(gen_embs, gt_embs_error.T)

            # 8. 找出 Gen 侧 4 句话中，与 Target GT 句子相似度最高的句子（作为 Query 的代表）
            # 针对 Gen 的 4 句话，求它们与 Target 5 句话的最大相似度的平均值（这就是 Avg-of-Max 的 Query 代表）
            # 简化操作：我们直接将 Gen 的 4 句文本拼接到一起作为 Query 的文本
            query_gen_text = " || ".join(self.gen_captions[query_img])

            # 9. 找到 Top-k_sentences (k=5) 的 GT 句子
            # 目标：找到 Gen 侧所有句子 vs. Target 侧所有句子的相似度中的 Top-K
            # 将 (4, 5) 矩阵展平为 (20,)
            flat_sims = pair_sim_matrix.flatten()
            flat_indices = np.argsort(-flat_sims) # 降序排序，找到 Top-K 的索引
            
            # 存储 Top-K 句子
            top_k_sentences = []
            for rank in range(min(k_sentences, len(flat_sims))):
                flat_idx = flat_indices[rank]
                
                # 展平索引转回 (gen_idx, gt_idx)
                gen_idx = flat_idx // pair_sim_matrix.shape[1] # 4 * 5 = 20
                gt_idx = flat_idx % pair_sim_matrix.shape[1]
                
                sim_score = flat_sims[flat_idx]
                gt_sentence = gt_captions_error[gt_idx]
                gen_sentence = self.gen_captions[query_img][gen_idx]
                
                top_k_sentences.append(f"[{sim_score:.4f}] Gen{gen_idx+1} vs. GT{gt_idx+1}: {gt_sentence}")

            # 10. 组装行数据
            row = [
                query_img,
                top1_retrieved_img,
                float(top1_score),
                query_gen_text
            ]
            row.extend(top_k_sentences)
            rows.append(row)

        # 11. 保存到 CSV
        header = ["query_img", "top1_retrieved_img", "top1_img_score", "query_gen_text"]
        header.extend([f"top{j+1}_error_sentence" for j in range(k_sentences)])
        
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"[INFO] 错误召回案例（Top-{k_images} 外，附带 {k_sentences} 句错误匹配）已导出到: {save_path}")

    def analyze_scores(self, k=5):

        """
        对正确召回和错误召回样本的 Top-1 相似度分数进行统计分析。
        前提：evaluate_avg_sim_i2t(k) 必须先运行。
        """
        if not self.retrieval_results:
            print("[ERROR] 请先运行 evaluate_avg_sim_i2t() 以生成检索结果。")
            return

        correct_scores = []
        error_scores = []

        for _, res_data in self.retrieval_results.items():
            top1_score = res_data['top1_score']
            if res_data['is_correct']:
                correct_scores.append(top1_score)
            else:
                error_scores.append(top1_score)

        correct_scores = np.array(correct_scores)
        error_scores = np.array(error_scores)
        
        # 统计结果
        stats = {
            f"Correct_in_Top{k}": {
                "Count": len(correct_scores),
                "Mean Score": correct_scores.mean() if correct_scores.size > 0 else 0.0,
                "Std Dev": correct_scores.std() if correct_scores.size > 0 else 0.0,
                "Median Score": np.median(correct_scores) if correct_scores.size > 0 else 0.0,
                "Min Score": correct_scores.min() if correct_scores.size > 0 else 0.0,
                "Max Score": correct_scores.max() if correct_scores.size > 0 else 0.0,
            },
            f"Error_out_of_Top{k}": {
                "Count": len(error_scores),
                "Mean Score": error_scores.mean() if error_scores.size > 0 else 0.0,
                "Std Dev": error_scores.std() if error_scores.size > 0 else 0.0,
                "Median Score": np.median(error_scores) if error_scores.size > 0 else 0.0,
                "Min Score": error_scores.min() if error_scores.size > 0 else 0.0,
                "Max Score": error_scores.max() if error_scores.size > 0 else 0.0,
            }
        }
        
        return stats
        

    def _prepare_export_rows(self, k, filter_correct):
        """ 准备正确/错误表格的行数据 """
        if not self.retrieval_results:
            print("[ERROR] 请先运行 evaluate_avg_sim_i2t() 以生成检索结果。")
            return []

        rows = []
        header = ["query_img"]
        # 动态创建 Top-K 的列名
        for j in range(k):
            header.append(f"retrieved_img_top{j+1}")
            header.append(f"score_top{j+1}")

        for query_img, res_data in self.retrieval_results.items():
            if res_data['is_correct'] != filter_correct:
                continue 

            row = [query_img]
            # 遍历 Top-K 结果 (已存储在 'top_k' 中)
            for j in range(k):
                if j < len(res_data['top_k']):
                    top_img, top_score = res_data['top_k'][j]
                else:
                    top_img, top_score = "N/A", "N/A" # 处理 K 超过实际结果数的情况
                
                row.append(top_img)
                row.append(float(top_score) if isinstance(top_score, float) else top_score)

            rows.append(row)
        
        return header, rows
    
    def export_correct_top_k_images(self, save_path, k=5):
        """ 导出 Top-K 内成功召回的样本，并列出 Top-K 结果 """
        header, rows = self._prepare_export_rows(k, filter_correct=True)

        if not rows:
            print(f"[INFO] 没有 {k} 个正确召回的样本可以导出。")
            return

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"[INFO] 正确召回案例 (Top-{k} 内, 包含 Top-{k} 结果) 已导出到: {save_path}")

    def export_errors_top_k_images(self, save_path, k=5):
        """ 导出 Top-K 内未能成功召回的样本，并列出 Top-K 结果 (错误的匹配) """
        header, rows = self._prepare_export_rows(k, filter_correct=False)
        
        if not rows:
            print(f"[INFO] 没有 {k} 个错误召回的样本可以导出。")
            return

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"[INFO] 错误召回案例 (Top-{k} 外, 包含 Top-{k} 结果) 已导出到: {save_path}")

    def _prepare_t2i_export_rows(self, k, target_recall_status, filter_correct):
        """ 
        准备 T2I 导出表格的行数据 (基于 1000 个图像的召回状态) 
        target_recall_status: 从 _compute_recall_5000_t2i 获得的 1000 图像的召回状态字典。
        """
        if not hasattr(self, 'retrieval_results_t2i'):
            print("[ERROR] 请先运行 evaluate_t2i() 以生成 T2I 检索结果。")
            return [], []

        rows = []
        header = ["query_img"]
        for j in range(k):
            header.append(f"retrieved_img_top{j+1}")
            header.append(f"score_top{j+1}")

        # 重新获取 1000 张图像的 R@K 状态，用 _compute_recall_5000_t2i 的 R@K 结果
        r1, r5, r10, image_recalled_status = self._compute_recall_5000_t2i(
            self.last_sim_matrix,
            self.gt_index_map,
            self.gen_img_list,
            return_status=True # ⚠️ 需要修改 _compute_recall_5000_t2i 以返回状态
        )
        
        # 遍历 1000 张图像 ID
        for query_img in self.gt_img_list: 
            # 判断是否成功召回 (以 R@K 为准)
            is_recalled = image_recalled_status.get(query_img, False)

            if is_recalled != filter_correct:
                continue 

            # 获取该图像最好的 Top-1 Score 对应的 Top-K 结果
            res_data = self.retrieval_results_t2i.get(query_img, {})

            row = [query_img]
            # 遍历 Top-K 结果 (已存储在 'top_k' 中)
            for j in range(k):
                if 'top_k' in res_data and j < len(res_data['top_k']):
                    top_img, top_score = res_data['top_k'][j]
                else:
                    top_img, top_score = "N/A", "N/A"
                
                row.append(top_img)
                row.append(float(top_score) if isinstance(top_score, float) else top_score)

            rows.append(row)
        
        return header, rows
    
    def export_t2i_correct_top_k_images(self, save_path, k=5):
        """ 导出 T2I Top-K 内成功召回的样本，并列出 Top-K 结果 """
        header, rows = self._prepare_t2i_export_rows(k, target_recall_status='R@K', filter_correct=True) # 使用 R@K 状态

        if not rows:
            print(f"[INFO] T2I 没有 {k} 个正确召回的样本可以导出。")
            return

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"[INFO] T2I 正确召回案例 (Top-{k} 内, 包含 Top-{k} 结果) 已导出到: {save_path}")

    def export_t2i_errors_top_k_images(self, save_path, k=5):
        """ 导出 T2I Top-K 内未能成功召回的样本，并列出 Top-K 结果 (错误的匹配) """
        header, rows = self._prepare_t2i_export_rows(k, target_recall_status='R@K', filter_correct=False) # 使用 R@K 状态
        
        if not rows:
            print(f"[INFO] T2I 没有 {k} 个错误召回的样本可以导出。")
            return

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"[INFO] T2I 错误召回案例 (Top-{k} 外, 包含 Top-{k} 结果) 已导出到: {save_path}")

    def analyze_t2i_scores(self, k=5):
        """
        对 T2I 检索结果 (1000张图像) 的 Top-1 相似度分数进行统计分析。
        Top-1 Score 取自该图像 5 个 Query 句子中的最高 Top-1 Score。
        召回状态 (Correct/Error) 取自 R@K (5句中只要有一句能召回就算 Correct)。
        """
        if not hasattr(self, 'retrieval_results_t2i'):
            print("[ERROR] 请先运行 evaluate_t2i() 以生成 T2I 检索结果。")
            return {}

        correct_scores = []
        error_scores = []

        # 重新获取 1000 张图像的 R@K 状态
        r1, r5, r10, image_recalled_status = self._compute_recall_5000_t2i(
            self.last_sim_matrix,
            self.gt_index_map,
            self.gen_img_list,
            return_status=True # ⚠️ 需要修改 _compute_recall_5000_t2i 以返回状态
        )
        
        for query_img in self.gt_img_list:
            res_data = self.retrieval_results_t2i.get(query_img, None)
            
            if res_data is None: continue

            # R@K 的召回状态
            is_recalled = image_recalled_status.get(query_img, False)
            # Top-1 Score (该图像 5 句中最高的 Top-1 Score)
            top1_score = res_data.get('top1_score', 0.0)

            if is_recalled:
                correct_scores.append(top1_score)
            else:
                error_scores.append(top1_score)

        correct_scores = np.array(correct_scores)
        error_scores = np.array(error_scores)
        
        # 统计结果
        # ... (与 analyze_scores 逻辑相同，省略)
        stats = {
            f"Correct_in_Top{k}": {
                "Count": len(correct_scores),
                "Mean Score": correct_scores.mean() if correct_scores.size > 0 else 0.0,
                "Std Dev": correct_scores.std() if correct_scores.size > 0 else 0.0,
                "Median Score": np.median(correct_scores) if correct_scores.size > 0 else 0.0,
                "Min Score": correct_scores.min() if correct_scores.size > 0 else 0.0,
                "Max Score": correct_scores.max() if correct_scores.size > 0 else 0.0,
            },
            f"Error_out_of_Top{k}": {
                "Count": len(error_scores),
                "Mean Score": error_scores.mean() if error_scores.size > 0 else 0.0,
                "Std Dev": error_scores.std() if error_scores.size > 0 else 0.0,
                "Median Score": np.median(error_scores) if error_scores.size > 0 else 0.0,
                "Min Score": error_scores.min() if error_scores.size > 0 else 0.0,
                "Max Score": error_scores.max() if error_scores.size > 0 else 0.0,
            }
        }
        
        return stats



    def save_sim_matrix(self, output_dir="output"):
        """
        保存最近一次计算的相似度矩阵及其行列对应的 Image ID。
        保存为两个文件:
        1. similarity_matrix.npy (纯数值矩阵)
        2. matrix_indices.json (包含 rows 和 cols 的图像ID列表)
        """
        import os
        
        if self.last_sim_matrix is None:
            print("[ERROR] 矩阵为空，请先运行 evaluate_i2t() 或相关评估函数。")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 保存矩阵 (.npy 格式最快且不丢失精度)
        npy_path = os.path.join(output_dir, "final_similarity_matrix.npy")
        np.save(npy_path, self.last_sim_matrix)
        
        # 2. 保存索引信息 (JSON 格式)
        # rows 对应 Gen/Query, cols 对应 GT/Target
        index_info = {
            "row_ids_query": self.last_query_list,   # 对应矩阵的行 (Query)
            "col_ids_target": self.last_target_list, # 对应矩阵的列 (Target)
            "shape": self.last_sim_matrix.shape
        }
        
        json_path = os.path.join(output_dir, "matrix_indices.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(index_info, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 相似度矩阵已保存至: {npy_path}")
        print(f"[INFO] 矩阵索引信息已保存至: {json_path}")



# Main
if __name__ == "__main__":

    print("Running with cumulative GT embedding logic...")
    TOP_K_IMAGES = 5
    try:
        # 请确保您的数据集路径正确
        evaluator = CaptionRetrievalEvaluatorFast(
            # gt_path="datasets/test_flickr30k.json",
            gt_path="datasets/val_coco2017.json",#COCO
            gen_path="output/COCO_captions_p5.json",#Flickr best: p5, p3
            # gen_path="output/flickr_best.json",
            model_name = "Model/bge-large-en-v1.5"
        )

        # Image-to-Text Retrieval
        r1, r5, r10 = evaluator.evaluate_i2t()
        print("====== I2T (Cumulative GT) ======")
        print(f"R@1 = {r1:.4f}")
        print(f"R@5 = {r5:.4f}")
        print(f"R@10 = {r10:.4f}")

        # === 2. 导出正确和错误表格（图像级别 Top-K）===
        # 正确表格：包含 Top-K 结果 (Query 目标是自身，通常是 Top-1)
        # evaluator.export_correct_top_k_images(f"output/coco/i2t_correct_top{TOP_K_IMAGES}_images.csv", k=TOP_K_IMAGES)
        # 错误表格：包含 Top-K 结果 (Query 目标不在 Top-K 内)
        # evaluator.export_errors_top_k_images(f"output/coco/i2t_errors_top{TOP_K_IMAGES}_images.csv", k=TOP_K_IMAGES)

        # === 3. 相似度分数统计分析 ===
        score_stats = evaluator.analyze_scores(k=TOP_K_IMAGES)
        print("\n====== Top-1 Similarity Score Analysis ======")
        for key, value in score_stats.items():
            print(f"--- {key} (Top-{TOP_K_IMAGES} boundary) ---")
            print(f"  Count: {value['Count']}")
            print(f"  Mean Top-1 Score: {value['Mean Score']:.4f}")
            print(f"  Median Top-1 Score: {value['Median Score']:.4f}")
            print(f"  Std Dev: {value['Std Dev']:.4f}")
            print(f"  Range: [{value['Min Score']:.4f}, {value['Max Score']:.4f}]")
        # Save I2T similarity matrix
        evaluator.save_sim_matrix("output/coco15/matrix_output_i2t")



        # ——————Text-to-Image Retrieval——————

        # r1, r5, r10 = evaluator.evaluate_max_sim_t2i_5000()
        # print("====== T2I (5000 Cumulative GT Sentences Query) ======")
        # print(f"R@1 = {r1:.4f}")
        # print(f"R@5 = {r5:.4f}")
        # print(f"R@10 = {r10:.4f}")
        
        r1, r5, r10 = evaluator.evaluate_t2i()
        print("====== T2I (5000 Cumulative GT Sentences Query) ======")
        print(f"R@1 = {r1:.4f}")
        print(f"R@5 = {r5:.4f}")
        print(f"R@10 = {r10:.4f}")
        # T2I 正确表格
        # evaluator.export_t2i_correct_top_k_images(f"output/coco/t2i_correct_top{TOP_K_IMAGES}_images.csv", k=TOP_K_IMAGES)
        # T2I 错误表格
        # evaluator.export_t2i_errors_top_k_images(f"output/coco/t2i_errors_top{TOP_K_IMAGES}_images.csv", k=TOP_K_IMAGES)
        # === T2I 相似度分数统计分析 ===
        t2i_score_stats = evaluator.analyze_t2i_scores(k=TOP_K_IMAGES)
        print("\n====== T2I Top-1 Similarity Score Analysis ======")
        for key, value in t2i_score_stats.items():
            print(f"--- {key} (Top-{TOP_K_IMAGES} boundary) ---")
            print(f"  Count: {value['Count']}")
            print(f"  Mean Top-1 Score: {value['Mean Score']:.4f}")
            print(f"  Median Top-1 Score: {value['Median Score']:.4f}")
            print(f"  Std Dev: {value['Std Dev']:.4f}")
            print(f"  Range: [{value['Min Score']:.4f}, {value['Max Score']:.4f}]")

        # Save T2I similarity matrix
        evaluator.save_sim_matrix("output/coco15/matrix_output_t2i")
        
        # evaluator.export_errors("output/i2t_errors.csv")
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. Please check paths: {e}")