'''
NOTE: This is a partial/lite version of our research code.
The full implementation for reproducing all experimental results 
will be made public upon paper acceptance.
'''
import os
import torch
import gc
import json
import time
from datetime import datetime


from stage1_captioning import run_captioning 
from stage2_HCE import CaptionRetrievalEvaluatorFast
from stage3_i2t import QwenReranker
from stage3_t2i import QwenRerankerT2I

def clear_gpu_memory():
    """clear GPU memory to prevent OOM issues"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def main():
    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # path configurations
        "gt_path": "datasets/gt.json",# GT JSON file path
        "image_root": "data/flickr",
        "output_base": "output",
        
        # model configurations
        "model_stage1": "Qwen/Qwen3-VL-4B-Instruct",
        # "model_stage2": "Model/bge-large-en-v1.5",
        'model_stage2': "Model/bge",# Word embedding model
        "model_stage3": "Qwen/Qwen3-VL-4B-Instruct",
        
        "top_k_images": 5,

        # experiment parameters
        # "top_k_coarse": 10,  
        "top_k_rerank": 5,   # Top-K
    }

    # auto-generated intermediate paths
    gen_caption_path = os.path.join(CONFIG["output_base"], "generated_captions_p5.json")
    matrix_output_i2t = os.path.join(CONFIG["output_base"], "matrix_output_i2t")
    matrix_output_t2i = os.path.join(CONFIG["output_base"], "matrix_output_t2i")

    os.makedirs(CONFIG["output_base"], exist_ok=True)

    #  Step 1: Visual Semantic Decomposition
    print("\n--- [Stage 1: Caption Generation] ---")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Stage1 Start Time: {now}")
    # run_captioning("data/testflickr", "prompt.txt", gen_caption_path)
    clear_gpu_memory() # free the memory


    # Step 2: HCE Strategy
    TOP_K_IMAGES = CONFIG["top_k_images"]
    print("\n--- [Stage 2: Coarse Retrieval & Evaluation] ---")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Stage2 Start Time: {now}")
    evaluator = CaptionRetrievalEvaluatorFast(
        gt_path=CONFIG["gt_path"],
        gen_path=gen_caption_path,
        model_name=CONFIG["model_stage2"]
    )

    # Run I2T evaluation and save matrix (for I2T reranking)
    r1, r5, r10 = evaluator.evaluate_i2t()
    print("====== I2T (Coarse) ======")
    print(f"R@1 = {r1:.4f}")
    print(f"R@5 = {r5:.4f}")
    print(f"R@10 = {r10:.4f}")
    # Save I2T similarity matrix
    # evaluator.save_sim_matrix(matrix_output_i2t)
    evaluator.save_sim_matrix("output/matrix_output_i2t")

    
    # Run T2I evaltion and save matrix (for T2I reranking)
    r1, r5, r10 = evaluator.evaluate_t2i()
    print("====== T2I (5000 Cumulative GT Sentences Query) ======")
    print(f"R@1 = {r1:.4f}")
    print(f"R@5 = {r5:.4f}")
    print(f"R@10 = {r10:.4f}")

    # Save T2I similarity matrix
    evaluator.save_sim_matrix("output/matrix_output_t2i")
    # evaluator.save_sim_matrix(matrix_output_t2i)
    
    del evaluator
    clear_gpu_memory() # free the memory

    # Step 3: I2T Reranking
    print("\n--- [Stage 3_1: I2T Reranking] ---")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Stage3_1 Start Time: {now}")
    reranker_i2t = QwenReranker(
        output_dir=matrix_output_i2t,
        gt_path=CONFIG["gt_path"],
        image_root=CONFIG["image_root"],
        model_path=CONFIG["model_stage3"],
        top_k=CONFIG["top_k_rerank"]
    )
    reranker_i2t.run_reranking()
    # reranker_i2t.run_stability_test_i2t(n_runs=5, subset_size=100)
    
    del reranker_i2t
    clear_gpu_memory()

    # Step 4: T2I Reranking
    print("\n--- [Stage 3_2: T2I Reranking] ---")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Stage3_2 Start Time: {now}")
    reranker_t2i = QwenRerankerT2I(
        output_dir=matrix_output_t2i,
        gt_path=CONFIG["gt_path"],
        image_root=CONFIG["image_root"],
        model_path=CONFIG["model_stage3"],
        top_k=CONFIG["top_k_rerank"]
    )
    reranker_t2i.run_reranking()
    # reranker_t2i.run_stability_test_t2i(n_runs=5, subset_size=100)
    
    del reranker_t2i
    clear_gpu_memory()

    print("\nThe experiment pipeline has completed successfully! All results are saved in the output directory.")

if __name__ == "__main__":
    main()
