import os
import faiss
import torch
import numpy as np
import argparse
import ipdb

# EMB_PATH = 'emb/objaverse_origin_img_emb_22w.pt'
# EMB_PATH = 'emb/lowpoly_origin_img_emb_building_22views.pt'
# FAISS_GPU_IDX_PATH = "emb/lowpoly_faiss_gpu_index_normalised_building_22views.npy"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', default='/your/emb/path/emb.pt')

    parser.add_argument('--faiss_output_name', default='/your/emb/path/faiss_idx_gltf_all.npy"')
    args = parser.parse_args()
    EMB_PATH = 'emb/lowpoly_origin_img_emb_building_22views.pt'
    FAISS_GPU_IDX_PATH = args.faiss_output_name
    EMB_PATH = args.emb_path


    image_embeds = torch.load(EMB_PATH)

    norms = torch.norm(image_embeds, p=2, dim=1, keepdim=True)
    normalised_img_embeds = image_embeds / norms
    normalised_img_embeds.shape

    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    faiss_gpu_index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), normalised_img_embeds.shape[1], config)
    # ipdb.set_trace()
    faiss_gpu_index.add(normalised_img_embeds.cpu().numpy().astype(np.float32))
    
    print(f"faiss_gpu_index.ntotal = {faiss_gpu_index.ntotal}")
    faiss_cpu_index = faiss.index_gpu_to_cpu(faiss_gpu_index)
    
    os.makedirs("emb", exist_ok=True)
    chunk = faiss.serialize_index(faiss_cpu_index)
    np.save(FAISS_GPU_IDX_PATH, chunk)
    print(f"{FAISS_GPU_IDX_PATH} saved")
    del faiss_gpu_index, faiss_cpu_index
