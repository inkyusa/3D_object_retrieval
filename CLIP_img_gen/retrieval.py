import os
import argparse
import faiss
import torch
import numpy as np
import pickle
from collections import Counter

import clip
import cv2
from PIL import Image

import ipdb

FAISS_GPU_IDX_PATH = "emb/lowpoly_faiss_gpu_index_normalised_building_22views.npy"
OBJ_PATH = 'emb/lowpoly_keys_building_22views.pkl'
EMB_2_OBJ = 'emb/emb_idx_2_lowpolyobj_idx_building_22views.pkl'

def load_faiss_index_gpu(path=FAISS_GPU_IDX_PATH):
    res = faiss.StandardGpuResources()
    faiss_gpu_index = faiss.deserialize_index(np.load(path))
    faiss_gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_gpu_index)
    
    return faiss_gpu_index

def load_emb_2_index(path=EMB_2_OBJ):
    with open(EMB_2_OBJ, 'rb') as file:
        emb2obj = pickle.load(file)
    return emb2obj

def load_obj_keys(path=OBJ_PATH):
    with open(OBJ_PATH, 'rb') as file:
        obj_keys = pickle.load(file)
    return obj_keys

def prepare_img(img):
    if img.max() <= 1.0:
        img = (img * 255).astype('uint8')

    img = Image.fromarray(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img

def calc_img_emb(img, model, preprocess, device):
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embed = model.encode_image(image_input)
    image_embed_norm = image_embed / image_embed.norm()
    
    return image_embed_norm

def calc_txt_emb(text, model, device):
    text_inputs = torch.cat([clip.tokenize(text)]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features = text_features / text_features.norm()
    
    return text_features
        

def retrive_topk_paths(faiss_gpu_index, image_embed_norm, emb2obj, obj_keys, objaverse_path, topk):
    distances, indices = faiss_gpu_index.search(
        np.array(image_embed_norm.cpu()).astype(np.float32),
        topk)
    
    key_idxes = []
    for idx in indices[0]:
        key_idxes.append(emb2obj[idx])
    idx_count = Counter(key_idxes)

    found_keys = []
    for idx in idx_count.most_common():
        # ipdb.set_trace()
        key = obj_keys[idx[0]]
        path = os.path.join(objaverse_path, f'{key}.gltf')
        found_keys.append(path)
    
    return found_keys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='data/mario.png')
    parser.add_argument("--topk", default=5)
    parser.add_argument("--objaverse_path", default='')

    args, extra = parser.parse_known_args()
    
    faiss_gpu_index = load_faiss_index_gpu()
    emb2obj = load_emb_2_index()
    obj_keys = load_obj_keys()

    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
    print(f"index.is_trained = {faiss_gpu_index.is_trained}")
    print(f"index.ntotal = {faiss_gpu_index.ntotal}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14')
    model.to(device)
    print('finished loading CLIP')
    
    # img = cv2.imread(args.input)
    # img = prepare_img(img)
    # image_embed_norm = calc_img_emb(img, model, preprocess, device)
    # paths = retrive_topk_paths(
    #     faiss_gpu_index, image_embed_norm,
    #     emb2obj, obj_keys, args.objaverse_path,
    #     args.topk)
    
    # for path in paths:
    #     print(path)
    
    # prompt = "old house"
    prompt = "red car"
    # text_features = calc_txt_emb("a photo of a cat", model, device)
    text_features = calc_txt_emb(prompt, model, device)
        
    paths = retrive_topk_paths(
        faiss_gpu_index, text_features,
        emb2obj, obj_keys, args.objaverse_path,
        args.topk)
    
    print(f"Input prompt is {prompt}")
    for path in paths:
        print(path)
