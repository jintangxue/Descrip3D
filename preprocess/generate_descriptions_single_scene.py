import os
import sys
import json
import torch
import argparse
import io
from PIL import Image
from filelock import FileLock
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

def is_near_edge(image, margin=30):
    width, height = image.size
    return width < 100 or height < 100

def run_scene(scene_id, json_path, image_root, output_file, model_path):
    lock_path = output_file + ".lock"

    skip_scene = False
    if os.path.exists(output_file):
        with FileLock(lock_path):
            try:
                with open(output_file, "r") as f:
                    output_data = json.load(f)
                if scene_id in {entry["scene_id"] for entry in output_data}:
                    print(f"[SKIP] Scene {scene_id} already processed.")
                    skip_scene = True
            except json.JSONDecodeError:
                pass
    if skip_scene:
        return

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    model.to(torch.device("cuda"))

    with open(json_path, "r") as f:
        image_to_objects = json.load(f)

    new_entries = []
    seen_ids = set()

    for img_path, data in image_to_objects.items():
        object_names = data["object_names"]
        object_ids = data["object_ids"]
        key_object_ids = data.get("key_object_ids", [])
        key_object_names = data.get("key_object_names", [])

        parts = img_path.strip().split("/")
        filename = parts[-1]
        img_path_proj = os.path.join(image_root, scene_id, filename)
        # img_path_proj = os.path.join(image_root, scene_id, "color", filename)

        if not os.path.exists(img_path_proj):
            continue

        try:
            image = Image.open(img_path_proj).convert("RGB")
            if is_near_edge(image):
                continue
        except:
            continue

        for obj_id, obj_name in zip(key_object_ids, key_object_names):
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)

            other_objs = [name for i, name in enumerate(object_names) if object_ids[i] != obj_id]
            other_objs_str = ", ".join(set(other_objs))

            question = (
                f"Describe clearly and briefly the relationships between the {obj_name} in the scene and nearby objects ({other_objs_str}). Do not describe objects you cannot see. Do not describe green labels."
            )

            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": question,
                "conv_mode": None,
                "image_file": img_path_proj,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()

            sys.stdout = io.StringIO()
            try:
                eval_model(args)
                response = sys.stdout.getvalue().strip()
            except Exception as e:
                response = f"[ERROR] {str(e)}"
            finally:
                sys.stdout = sys.__stdout__

            new_entries.append({
                "scene_id": scene_id,
                "object_id": obj_id,
                "object_name": obj_name,
                "object_description": response if response else "[ERROR] No response"
            })
            
            print(f"[img_path_proj] {img_path_proj}")
            print(f"[Object_description] {response}")
            print(f"[Other_object] {other_objs_str}")
            print(f"[SAVED] {scene_id} - {obj_name} ({obj_id})")

    try:
        with FileLock(lock_path):
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    try:
                        output_data = json.load(f)
                    except json.JSONDecodeError:
                        output_data = []
            else:
                output_data = []

            existing_keys = {(entry["scene_id"], entry["object_id"]) for entry in output_data}
            for entry in new_entries:
                key = (entry["scene_id"], entry["object_id"])
                if key not in existing_keys:
                    output_data.append(entry)

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)

        print(f"[SAVED] {scene_id}: {len(new_entries)} new descriptions")

    except Exception as e:
        print(f"[ERROR] Failed to save: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    args = parser.parse_args()

    run_scene(args.scene_id, args.json_path, args.image_root, args.output_file, args.model_path)