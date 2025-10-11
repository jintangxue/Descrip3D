# gpu_scheduler.py
import os
import json
import subprocess
import time
import queue
from threading import Thread
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output-file", default="./object_descriptions_all_scenes.json")
    p.add_argument("--model", default="liuhaotian/llava-v1.5-7b")
    p.add_argument("--gpus", default="0,1")
    args = p.parse_args()

    json_dir = args.json_dir
    image_root = args.image_root
    output_file = args.output_file
    model_path = args.model
    available_gpus = [int(x) for x in args.gpus.split(",")]

    completed_scenes = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for entry in json.load(f):
                completed_scenes.add(entry["scene_id"])

    scene_map = {
        fname.split("_image_to_objects.json")[0]: os.path.join(json_dir, fname)
        for fname in os.listdir(json_dir)
        if fname.endswith("_image_to_objects.json")
    }

    scene_queue = queue.Queue()
    for scene_id, path in scene_map.items():
        if scene_id not in completed_scenes:
            scene_queue.put((scene_id, path))

    def gpu_worker(gpu_id):
        while not scene_queue.empty():
            try:
                scene_id, json_path = scene_queue.get_nowait()
            except queue.Empty:
                break

            print(f"[GPU {gpu_id}] Starting {scene_id}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = [
                "python", "generate_descriptions_single_scene.py",
                "--scene-id", scene_id,
                "--json-path", json_path,
                "--image-root", image_root,
                "--output-file", output_file,
                "--model-path", model_path
            ]

            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Error processing {scene_id}: {e}")
            finally:
                scene_queue.task_done()

    threads = []
    for gpu in available_gpus:
        t = Thread(target=gpu_worker, args=(gpu,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n All scenes processed and saved.")

