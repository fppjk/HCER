# HCER: Hierarchical Contextual Embeddings and Re-ranking for Training Free Cross-Modal Retrieval
HCER: Hierarchical Contextual Embeddings and Re-ranking for Training-Free Cross-Modal Retrieval

## Abstract

Generation-based training-free cross-modal retrieval leverages Multimodal Large Language Models (MLLMs) to map images into textual descriptions for unified matching. However, holistic embeddings often dilute fine-grained visual features due to semantic redundancy, the sparse distribution of atomic-level semantics within long sequences, and the inability of cosine similarity to verify complex cross-modal consistency. 

To overcome these limitations, we propose **HCER (Hierarchical Contextual Embeddings and Re-ranking)**, a fully training-free framework following a **"Decomposition-Hierarchy-Verification"** paradigm. 
1. **Decomposition**: Utilizes frozen MLLMs with structured prompts to transform visual inputs into multi-perspective textual descriptions, extracting explicit atomic-level semantic units. 
2. **Hierarchy**: Introduces Hierarchical Contextual Embeddings (HCE), which decouples these descriptions into an independent stream for local discriminative power and a joint stream for global contextual coherence, fusing them to prevent detail dilution. 
3. **Verification**: Employs an MLLM-based Semantic Consistency Reasoning Re-ranking module to perform explicit logical validation on Top-K candidates, moving beyond surface-level statistical similarity. 

Extensive experiments on **Flickr30K** and **MSCOCO** demonstrate that HCER significantly improves zero-shot retrieval performance, particularly in Recall@1, proving the efficacy of integrating atomic-level semantics with explicit reasoning without task-specific optimization.

---

## Framework

项目遵循“分解-分层-验证”范式，主要包含以下核心步骤：

![HCER Framework Architecture](此处放置你的架构图文件路径，例如：images/framework.png)
*图 1: HCER 整体架构图。*

- **Semantic Decomposition**: 利用结构化提示词将图像转化为多视角（Multi-view）原子级语义。
- **Hierarchical Contextual Embeddings (HCE)**: 解决长描述中的语义稀释问题，融合局部与全局特征。
- **Reasoning-based Re-ranking**: 利用 MLLM 对 Top-K 候选集进行显式的逻辑一致性推理验证。

---

## 实验结果 (Experimental Results)

### 1. Flickr30K 性能对比
在 Flickr30K 数据集上，HCER 在多项指标（尤其是 R@1）上达到了当前最优水平。

| Method | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIP (ViT-L) | 87.2 | 98.3 | 99.4 | 67.3 | 89.0 | 93.3 |
| BLIP (ViT-L) | 75.5 | 95.1 | 97.7 | 70.0 | 91.2 | 95.2 |
| VL-DE (2024) | 83.7 | 96.7 | 99.0 | 65.3 | 88.8 | 93.1 |
| D2E-VSE (2025) | 84.1 | 96.1 | 98.3 | 68.5 | 91.3 | 94.9 |
| E5V (2024) | 88.2 | 98.7 | 99.4 | 79.5 | **95.0** | **97.6** |
| WaveDN (2024)* | 82.3 | 96.3 | 97.8 | 63.7 | 87.2 | 93.0 |
| ImageScope (2025)* | 81.1 | 94.0 | 96.8 | - | - | - |
| LexiCLIP (2025)* | 92.9 | - | 97.1 | 79.2 | - | 97.4 |
| **HCER (Ours)** | **96.5** | **99.0** | **99.7** | **83.0** | 91.7 | 95.3 |

### 2. MSCOCO 性能对比
在 MSCOCO (5K test set) 上，HCER 同样展现了极强的零样本检索能力。

| Method | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIP (ViT-L) | 58.1 | 81.0 | 87.8 | 37.0 | 61.6 | 71.5 |
| BLIP (ViT-L) | 63.5 | 86.5 | 92.5 | 48.4 | 74.4 | 83.2 |
| VL-DE (2024) | 63.4 | 87.6 | **93.7** | 46.5 | 75.3 | 84.9 |
| D2E-VSE (2025) | 60.6 | 86.5 | 93.2 | 46.8 | 76.4 | **85.7** |
| E5V (2024) | 62.0 | 83.6 | 89.7 | 52.0 | **76.5** | 84.7 |
| WaveDN (2024)* | 50.9 | 75.4 | 83.7 | 31.4 | 56.5 | 67.6 |
| ImageScope (2025)* | 53.7 | 75.9 | 83.5 | - | - | - |
| LexiCLIP (2025)* | 67.4 | - | 92.1 | 52.7 | - | 84.5 |
| **HCER (Ours)** | **75.5** | **88.1** | 92.9 | **54.4** | 68.5 | 77.6 |

*\*Note: '*' indicates training-free models.*

---

## 可视化 (Visualization)

![Similarity Matrix](此处放置图14或矩阵图路径)
*图 2: HCE 编码嵌入与文本库之间的余弦相似度矩阵。*

![Candidate Selection](此处放置Table IX/X对应的可视化效果图路径)
*图 3: 检索示例及推理重排前后的对比。*

---

## 开源声明 (Availability)

**The source code will be made publicly available upon the acceptance of the paper.** (代码将在论文被接收后正式开源。)
