# Faiss toy example and profiling

This repo contains a toy faiss example that retrieves images from unsplash25k dataset as shown below

<img src="./assets/diagram.png" width="700">

## Software version
| Package | Version      | Description            |
|---------|--------------|------------------------|
| python  | 3.8.13      |        |
| faiss   | 1.7.2   |  |
| torch   | 1.13.0+cu117   |   |
| torchvision   | 0.14.0   |   |
| gradio   | 4.4.1  |   |
| diffusers | 0.23.1 |    |
| CUDNN | 8500| |
| CUDA version| 11.7| |
- Note that faiss-gpu is sensitive to dependencies' version. If you encounter getting stuck or stalling more than a minute while operating faiss-gpu, please make sure package versions.

## Dataset download
One can download the public dataset from [unsplash25k from hf](https://huggingface.co/datasets/jamescalam/unsplash-25k-photos) and save them under `input/unsplash` folder

## Technical detail and results
More technical detail and experimental results can be found from [here](https://tencentoverseas-my.sharepoint.com/:p:/g/personal/inkyusa_global_tencent_com/ESpkXrXoKTVLiNJRMkgX9_MBpTdQJ0-xG4eZOt6ENNRZDA?e=auxd90)

<img src="./assets/profiling.png" width="700">

# Objaverse Retrival

## 1. Calculate CLIP image embeddings by running

  `CLIP_img_gen/calc_clip_img_embedding.py`

## 2. Prepare the packed embedding in one single file

  `CLIP_img_gen/prepare_embeddings.py`

## 3. Generate faiss gpu index

  `CLIP_img_gen/gen_faiss_index_gpu.py`

## 4. Test retrive from objaverse

  `CLIP_img_gen/retrieval.py`

## 5. Deploy the webui

  ### 5.1 Run webui: `python webui/gradio_app.py`



