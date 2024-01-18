import os
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", default='/apdcephfs_cq3/share_2909871/inkyu/low_poly_retrieval/renderings_building_22views')
parser.add_argument("--output_file", default='/apdcephfs_cq3/share_2909871/inkyu/low_poly_retrieval/rendering_folder_names_building_22views.json')
args, extra = parser.parse_known_args()
print(f"args={args}")
def generate_json_from_subfolders(folder_path, output_file):
    all_items = os.listdir(folder_path)
    directories = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]
    with open(output_file, 'w') as file:
        json.dump(directories, file, indent=4)
generate_json_from_subfolders(args.folder_path, args.output_file)