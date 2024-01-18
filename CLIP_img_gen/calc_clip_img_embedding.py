# import OpenEXR
import torch
import clip
from PIL import Image
import argparse
import cv2
import os
from pathlib import Path
import pandas as pd
import numpy as np
from filelock import FileLock
import random
import json
# import Imath
from tqdm import tqdm
import pdb

def rfind_idx(str, cnt, token='_'):
    """
    Reverse find the index of the `cnt`th token.
    Should be faster than str.split or regex, I guess...
    """
    for i in range(len(str) - 1, -1, -1):
        if str[i] == token:
            cnt -= 1

            if cnt == 0:
                return i
    
    return -1

def find_degree_chunk(filename):
    basename = os.path.splitext(filename)[0]
    sidx = rfind_idx(basename, 1, '_')
    return basename[sidx + 1:]

def get_azimuth_polor(chunk):
    return [int(c) for c in chunk.split('#')]

# def extract_channel(exr_file, channel_name):
#     """
#     Read specific channel data and return numpy array
#     """

#     channel_header = exr_file.header()['channels'][channel_name]
#     channel_data = exr_file.channel(channel_name, channel_header.type)
#     format = Imath.PixelType(channel_header.type.v)

#     if format.v == Imath.PixelType.FLOAT:
#         channel_np = np.frombuffer(channel_data, dtype=np.float32)
#     elif format.v == Imath.PixelType.HALF:
#         channel_np = np.frombuffer(channel_data, dtype=np.float16)
#     else:  # UINT
#         channel_np = np.frombuffer(channel_data, dtype=np.uint32)
#     return channel_np

# def read_exr(path):
#     file = OpenEXR.InputFile(path)

#     # get image size
#     dw = file.header()['dataWindow']
#     width, height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

#     # extract RGBA
#     red = extract_channel(file, 'R').reshape((height, width))
#     green = extract_channel(file, 'G').reshape((height, width))
#     blue = extract_channel(file, 'B').reshape((height, width))

#     channels = [red, green, blue]
#     if 'A' in file.header()['channels']:
#         alpha = extract_channel(file, 'A').reshape((height, width))
#         channels.append(alpha)
#     raw_rgba = cv2.merge(channels)

#     return raw_rgba

def read_img(path):
    img_bgr = cv2.imread(path)
    rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return rgb_image

def load_seen(path, lock_path):
    seen = set()
    if os.path.exists(path):
        lock = FileLock(lock_path)
        with lock:
            df = pd.read_csv(path)
            seen = set(df['key'])
    else:
        with open(path, 'w') as fo:
            fo.write('key,min,mean,max\n')
    
    return seen

def load_skip(args):
    skip = set()
    if args.skip_list is not None:
        with open(args.skip_list) as fi:
            for line in fi:
                skip.add(line.strip())
            
        print(f'loaded {len(skip)} skip keys')

    return skip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str, default='/apdcephfs_cq3/share_2909871/inkyu/low_poly_retrieval/renderings')
    # parser.add_argument("--input", default='/apdcephfs_cq3/share_2909871/kownseduan/code/text_to_shape_zero123/objaverse_noun_cnt_less_4_clip_02.json')
    # parser.add_argument("--input", default='/apdcephfs_cq3/share_2909871/kownseduan/code/text_to_shape_zero123/objaverse_noun_cnt_less_3.json')
    # parser.add_argument("--outdir", default='/apdcephfs_cq8/share_2909871/kownseduan/embedding/objaverse/image_clip_latent/version1')
    parser.add_argument("--input", default='/apdcephfs_cq3/share_2909871/inkyu/low_poly_retrieval/rendering_folder_names_building_22views.json')
    parser.add_argument("--outdir", default='/apdcephfs_cq3/share_2909871/inkyu/low_poly_retrieval/low_poly_embs_building_22views')

    args, extra = parser.parse_known_args()
    print(f"args={args}")
    
    keys = json.load(open(args.input))
    print(f'found {len(keys)} keys')


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14')
    model.to(device)

    random.shuffle(keys)
    with tqdm(total=len(keys)) as pbar:
        cnt = 0
        for sub in keys:
            pbar.update(1)
            
            outdir = os.path.join(args.outdir, sub)
            os.makedirs(outdir, exist_ok=True)

            full_sub = os.path.join(args.data_root, sub)
            directory_path = Path(full_sub)
            # bg_files = directory_path.rglob('bg_*.exr')
            bg_files = directory_path.rglob('bg_*.png')
            # pdb.set_trace()
            imgs = []
            for filepath in bg_files:
                filepath = os.path.basename(filepath)
                print(f"filepath={filepath}")
                out_path = os.path.join(outdir, f'{filepath}.clip.embedding')
                # if os.path.exists(out_path):
                #     continue
                
                chunk = find_degree_chunk(filepath)
                azimuth, polar = get_azimuth_polor(chunk)

                exr_path = os.path.join(full_sub, filepath)
                print(f"exr_path={exr_path}")
                try:
                    # img = read_exr(exr_path)
                    img = read_img(exr_path)
                except:
                    continue
                # pdb.set_trace()
                # img = (img * 255).astype('uint8')
                img = Image.fromarray(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                image_input = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = model.encode_image(image_input)

                with open(out_path, 'wb') as fo:
                    torch.save(image_features, fo)
            