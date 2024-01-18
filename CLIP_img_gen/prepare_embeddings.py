import os
import argparse
# import faiss
import torch
import pickle
import ipdb
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/apdcephfs_cq3/share_2909871/inkyu/low_poly_retrieval/low_poly_embs_building_22views')
    args = parser.parse_args()
    EMB_PATH = 'emb/lowpoly_origin_img_emb_building_22views.pt'
    OBJ_PATH = 'emb/lowpoly_keys_building_22views.pkl'
    EMB_2_OBJ_PATH = 'emb/emb_idx_2_lowpolyobj_idx_building_22views.pkl'

    
    objs = []
    embs = []
    embs_to_objs = {}
    
    dirs = os.listdir(args.data_path)
    for sub_dir in tqdm(dirs):
        # print(sub_dir)
        obj_path = os.path.join(args.data_path, sub_dir)
        
        objs.append(sub_dir)
        for emb_path in os.listdir(obj_path):
            if '.embedding' not in emb_path:
                continue
            
            emb_full_path = os.path.join(obj_path, emb_path)
            emb = torch.load(emb_full_path).to('cpu')
            # print(emb_path)
            
            embs.append(emb)
            embs_to_objs[len(embs) - 1] = len(objs) - 1
            
        # if len(embs) > 100:
        #     break
        
    print(len(embs))
    # ipdb.set_trace()
    all_embs = torch.concat(embs, dim=0)
    torch.save(all_embs, EMB_PATH)
    
    with open(OBJ_PATH, 'wb') as file:
        pickle.dump(objs, file)
        
    with open(EMB_2_OBJ_PATH, 'wb') as file:
        pickle.dump(embs_to_objs, file)
    