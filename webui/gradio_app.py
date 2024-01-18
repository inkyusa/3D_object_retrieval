import os
# os.environ['U2NET_HOME'] = '/root/.cache/.u2net'
# os.environ['http_proxy'] = 'http://star-proxy.oa.com:3128'
# os.environ['https_proxy'] = 'http://star-proxy.oa.com:3128'
# os.environ['no_proxy'] = 'localhost, 127.0.0.1, ::1'

import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)

import argparse
import gradio as gr
import torch

import cv2
import clip
import numpy as np
import rembg
import torch
from PIL import Image

OBJAVERSE_PATH = '/home/inkyu/workspace/gltf_all'
TMP_PNG = 'data/webui.png'

from CLIP_img_gen.retrieval import (
    load_faiss_index_gpu,
    load_emb_2_index,
    load_obj_keys,
    prepare_img,
    calc_img_emb,
    calc_txt_emb,
    retrive_topk_paths,
)

from ipdb import set_trace as st
    
def update_progress(progress, progress_path):
    with open(progress_path) as fi:
        prog = fi.read()
        # print(prog)
        try:
            title, ratio = prog.strip().split(' ')
            title = title.replace('_', ' ')
            ratio = float(ratio)
            progress(ratio, desc=title)
        except:
            print(f'can not parse progress: {prog}')

from diffusers import StableDiffusionXLPipeline
def load_SDXL_pipeline():
    print('load SDXL')
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "/root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/f898a3e026e802f68796b95e9702464bac78d76f", 
        # torch_dtype=torch.float16,
        revision="f898a3e02"
    )
    pipe.to(torch_device="cuda")
    return pipe

faiss_gpu_index = load_faiss_index_gpu()
emb2obj = load_emb_2_index()
obj_keys = load_obj_keys()

# sdxl_pipe = load_SDXL_pipeline()
rembg_session = rembg.new_session()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14')
model.to(device)

        
def txt_img(
    prompt,
    negative_prompt,
):
    print(prompt)

    images = sdxl_pipe(prompt=prompt, negative_prompt=negative_prompt,output_type="pil").images

    images[0].save(TMP_PNG)
    images = None
    
    torch.cuda.empty_cache()
    yield TMP_PNG
    
def retrive(
    ref_image,
    progress=gr.Progress()
    ):
    
    progress(0, desc='removing background')
    
    img = cv2.imread(ref_image, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(TMP_PNG, img)
    
    tmp = Image.open(TMP_PNG).convert('RGBA')
    img = rembg.remove(tmp, session=rembg_session)
    
    # img = prepare_img(np.array(img))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    image_embed_norm = calc_img_emb(img, model, preprocess, device)
    paths = retrive_topk_paths(
        faiss_gpu_index, image_embed_norm,
        emb2obj, obj_keys,
        OBJAVERSE_PATH,
        10)
    
    while len(paths) < 3:
        paths.append(paths[0])

    return paths[:3]

def retrive_txt(prompt):
    query_txt_embed = calc_txt_emb(prompt, model, device)
    paths = retrive_topk_paths(
        faiss_gpu_index, query_txt_embed,
        emb2obj, obj_keys,
        OBJAVERSE_PATH,
        10)
    
    while len(paths) < 3:
        paths.append(paths[0])
        
    return paths[:3]

def launch(
    port,
    listen=False,
):
    lock_path = 'runing'

    global listen_port
    listen_port = port

    css = """
    #config-accordion, #logs-accordion {color: black !important;}
    .dark #config-accordion, .dark #logs-accordion {color: white !important;}
    .stop {background: darkred !important;}
    """

    with gr.Blocks(
        title="Objaverse Retrive - Web Demo",
        css=css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
            # Objaverse Retrive Demo
            """
            gr.Markdown(header)
        
        # with gr.Row(equal_height=True):
        #     with gr.Column(scale=2):
        #         with gr.Row(equal_height=True):
        #             prompt_input = gr.Textbox(
        #                 value='',
        #                 placeholder="Prompt",
        #                 show_label=False,
        #                 interactive=True)
        #         with gr.Row(equal_height=True):
        #             negative_prompt = gr.Textbox(
        #                 value='',
        #                 placeholder="Negative Prompt",
        #                 show_label=False,
        #                 interactive=True)
        #     with gr.Column(scale=1):
        #         txt2img_btn = gr.Button(value="Generate Image SDXL(slow)", variant="primary")
                
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    txt_input = gr.Textbox(
                        value='',
                        placeholder="Prompt",
                        show_label=False,
                        interactive=True)
            with gr.Column(scale=1):
                txt_btn = gr.Button(value="Retrive using Text", variant="primary")
            
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                ref_image = gr.Image(
                    value=None,
                    label="Reference Image",
                    type='filepath',
                    image_mode='RGB',
                    # tool='select',
                    interactive=True)
            
            with gr.Column(scale=1):
                with gr.Row():
                    retrive_btn = gr.Button(value="Retrive using Image", variant="primary")
                with gr.Row():
                    output_mesh1 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                with gr.Row():
                    output_mesh2 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                with gr.Row():
                    output_mesh3 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                
        # txt2img_btn.click(
        #     fn=txt_img,
        #     inputs=[
        #         prompt_input,
        #         negative_prompt,
        #     ],
        #     outputs=[
        #         ref_image
        #     ],
        #     concurrency_limit=1
        # )
        
        txt_btn.click(
            fn=retrive_txt,
            inputs=[
                txt_input
            ],
            outputs=[
                output_mesh1,
                output_mesh2,
                output_mesh3,
            ],
            concurrency_limit=1
        )

        retrive_btn.click(
            fn=retrive,
            inputs=[
                ref_image
            ],
            outputs=[
                output_mesh1,
                output_mesh2,
                output_mesh3,
            ],
            concurrency_limit=1
        )

    launch_args = {"server_port": port}
    if listen:
        launch_args["server_name"] = "0.0.0.0"
    demo.queue().launch(**launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    launch(
        args.port,
        listen=args.listen,
    )
