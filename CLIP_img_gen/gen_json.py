import os
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", default='/home/inkyu/workspace/objaverse-rendering-private/rendered/gltf_all')
parser.add_argument("--output_file", default='/home/inkyu/workspace/objaverse-rendering-private/rendered/rendering_folder_names_gltf_all.json')
args, extra = parser.parse_known_args()
print(f"args={args}")
def generate_json_from_subfolders(folder_path, output_file):
    all_items = os.listdir(folder_path)
    directories = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]
    with open(output_file, 'w') as file:
        json.dump(directories, file, indent=4)
generate_json_from_subfolders(args.folder_path, args.output_file)