import os
import cv2
import numpy as np
import imageio
from glob import glob
from plyfile import PlyData
from scipy.spatial import ConvexHull
import torch
import os
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from fusion_util import PointCloudToImageMapper
from scipy.spatial import ConvexHull
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from collections import defaultdict
import time
from plyfile import PlyData
import json
from itertools import combinations
from multiprocessing import Pool, cpu_count
import argparse


def visualize_instance_projection(image_path, projected_points_dict, object_names=None, save_path=None, show=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for inst_id, points in projected_points_dict.items():
        if len(points) < 3:
            continue
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            for i in range(len(hull_points)):
                pt1 = tuple(hull_points[i % len(hull_points)].astype(int))
                pt2 = tuple(hull_points[(i + 1) % len(hull_points)].astype(int))
            
            centroid = hull_points.mean(axis=0).astype(int)
            if object_names and inst_id < len(object_names):
                label_text = f"{object_names[inst_id]}"
            cv2.putText(image, label_text, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except:
            continue

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def process_scene_for_visualization(scene_id, data_path, image_root, mask3d_root, intrinsics_path, output_dir, depth_scale=1000.0):
    from fusion_util import PointCloudToImageMapper

    print(f"Start processing {scene_id}")

    plydata = PlyData.read(open(data_path, 'rb'))
    points = np.array([list(x) for x in plydata.elements[0]])
    locs_in = np.ascontiguousarray(points[:, :3])

    mask3d_data = torch.load(os.path.join(mask3d_root, f"{scene_id}.pth"))
    object_names = mask3d_data[2]
    inst_segids = mask3d_data[3]
    inst_num = len(inst_segids)
    n_points = locs_in.shape[0]

    intrinsics = np.loadtxt(intrinsics_path)
    img_dim = (240, 320)
    mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=intrinsics,
        visibility_threshold=0.25, cut_bound=10)

    scene_2d_dir = os.path.join(image_root, scene_id)
    img_dirs = sorted(glob(os.path.join(scene_2d_dir, 'color/*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))

    image_objects = defaultdict(lambda: {"object_names": [], "object_ids": [], "key_object_names": [], "key_object_ids": []})

    for img_id, img_dir in enumerate(img_dirs):
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        depthpath = img_dir.replace('color', 'depth').replace('.jpg', '.png')

        if not os.path.exists(posepath) or not os.path.exists(depthpath):
            continue

        pose = np.loadtxt(posepath)
        depth = imageio.v2.imread(depthpath) / depth_scale

        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = mapper.compute_mapping(pose, locs_in, depth)

        if mapping[:, 3].sum() == 0:
            continue

        mapping = torch.from_numpy(mapping)
        mask = mapping[:, 3]
        mask_ids = torch.arange(mask.shape[0])[mask.bool()].tolist()

        projected_points_dict = {}
        centroids = {}

        for instid in range(inst_num):
            inst_seg = inst_segids[instid]
            overlap_ids = list(set(mask_ids).intersection(set(inst_seg)))
            if overlap_ids:
                single_inst_points = mapping[overlap_ids][:, [2, 1]]
                if len(single_inst_points) >= 3:
                    projected_points_dict[instid] = single_inst_points.numpy()
                    centroids[instid] = single_inst_points.float().mean(dim=0).numpy()

        img_h, img_w = img_dim
        margin = 40
        for instid, centroid in centroids.items():
            x, y = centroid
            if margin <= x <= img_w - margin and margin <= y <= img_h - margin:
                image_objects[img_dir]["key_object_names"].append(object_names[instid])
                image_objects[img_dir]["key_object_ids"].append(instid)

        if projected_points_dict:
            filtered_projected_points = {}
            used_ids = set()
            centroids_by_name = defaultdict(list)

            for instid, points in projected_points_dict.items():
                centroid = points.mean(axis=0)
                name = object_names[instid]
                centroids_by_name[name].append((centroid, instid))

            distance_thresh = 20
            for name, centroids_group in centroids_by_name.items():
                selected = []
                for i, (c_i, id_i) in enumerate(centroids_group):
                    if id_i in used_ids:
                        continue
                    keep = True
                    for (c_j, id_j) in selected:
                        if np.linalg.norm(c_i - c_j) < distance_thresh:
                            keep = False
                            break
                    if keep:
                        selected.append((c_i, id_i))
                        used_ids.add(id_i)

            for instid in used_ids:
                filtered_projected_points[instid] = projected_points_dict[instid]
                image_objects[img_dir]["object_names"].append(object_names[instid])
                image_objects[img_dir]["object_ids"].append(instid)

            img_filename = os.path.basename(img_dir)
            save_path = os.path.join(output_dir, scene_id, img_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            visualize_instance_projection(img_dir, filtered_projected_points, object_names, save_path=save_path)

        else:
            print(f"⚠️ Skipping {img_dir}: no objects with enough overlap found.")

    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, f"{scene_id}_image_to_objects.json")
    with open(json_output_path, "w") as f:
        json.dump(image_objects, f, indent=4)
    print(f"✅ Saved object mapping to {json_output_path}")


##################### For All Scenes ########################################
def process_scene_wrapper(args):
    scene_id, ply_file, config = args
    try:
        process_scene_for_visualization(
            scene_id=scene_id,
            data_path=ply_file,
            image_root=config["image_root"],
            mask3d_root=config["mask3d_root"],
            intrinsics_path=config["intrinsics_path"],
            output_dir=config["output_dir"]
        )
    except Exception as e:
        print(f"❌ Failed to process {scene_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mask3d_root", type=str, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    config = {
        "image_root": args.image_root,
        "mask3d_root": args.mask3d_root,
        "intrinsics_path": args.intrinsics_path,
        "output_dir": args.output_dir
    }

    scan_folders = sorted(glob(os.path.join(args.scan_root, "scene*/")))
    tasks = []
    for scan_path in scan_folders:
        scene_id = os.path.basename(os.path.normpath(scan_path))
        ply_file = os.path.join(scan_path, f"{scene_id}_vh_clean_2.ply")
        if not os.path.exists(ply_file):
            print(f"⚠️ Skipping {scene_id}, .ply file not found.")
            continue

        output_json_path = os.path.join(args.output_dir, f"{scene_id}_image_to_objects.json")
        # if os.path.exists(output_json_path):
        #     print(f"✅ Skipping {scene_id}, already processed.")
        #     continue

        tasks.append((scene_id, ply_file, config))

    with Pool(processes=args.num_workers) as pool:
        pool.map(process_scene_wrapper, tasks)
