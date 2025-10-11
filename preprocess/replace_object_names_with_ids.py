import json
import os
import re
from glob import glob
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True, help="raw or cleaned descriptions JSON")
    ap.add_argument("--json-dir", required=True, help="folder with scene*_image_to_objects.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    desc_file = args.desc
    image_obj_dir = args.json_dir
    output_file = args.output

    with open(desc_file, 'r') as f:
        descriptions = json.load(f)

    scene_image_map = {}

    json_files = glob(os.path.join(image_obj_dir, "scene*_image_to_objects.json"))
    for file_path in json_files:
        scene_id = os.path.basename(file_path).split("_image_to_objects.json")[0]
        with open(file_path, 'r') as f:
            image_data = json.load(f)
            scene_image_map[scene_id] = image_data

    def replace_desc_with_id(scene_id, target_obj_id, object_description):
        if scene_id not in scene_image_map:
            return object_description

        for image_path, obj_data in scene_image_map[scene_id].items():
            if target_obj_id in obj_data["object_ids"]:
                replacement_map = {}
                for name, obj_id in zip(obj_data["object_names"], obj_data["object_ids"]):
                    tag = f"<OBJ{obj_id:03d}>"
                    replacement_map[name.lower()] = tag

                desc = object_description
                for name, tag in replacement_map.items():
                    desc = re.sub(rf'\b{re.escape(name)}\b', tag, desc, flags=re.IGNORECASE)
                return desc

        return object_description

    updated_descriptions = []
    for entry in descriptions:
        scene_id = entry["scene_id"]
        object_id = entry["object_id"]
        original_desc = entry["object_description"]
        new_desc = replace_desc_with_id(scene_id, object_id, original_desc)
        updated_entry = entry.copy()
        updated_entry["object_description"] = new_desc
        updated_descriptions.append(updated_entry)

    with open(output_file, 'w') as f:
        json.dump(updated_descriptions, f, indent=2)

    print(f"Updated descriptions written to {output_file}")