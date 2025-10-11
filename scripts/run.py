import os
import random
import subprocess
from datetime import datetime

which_python = subprocess.check_output("which python", shell=True).decode().strip()
os.environ['PYTHONPATH'] = f"{os.getenv('PYTHONPATH', '')}:{which_python}:."
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

os.environ['MASTER_PORT'] = str(54000 + random.randint(0, 9999))
os.environ['MASTER_ADDR'] = 'localhost'

epoch = 3
batch_size = 32
lr = 5e-6
train_emb = True
train_img_proj = True
add_img_token = True
add_scene_token = False
no_obj = False
input_dim = 1024
bidirection = False
different_lr = False
max_obj_num = 100
lora_r = 16
lora_alpha = 16
add_pos_emb = False
feat_fusion = False
fuse_with_id = False
config = ""
max_grad_norm = 0.01
seed = 42
use_location_token = False

llama_model_path = "llm/vicuna-7b-v1.5"

train_tag = "scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref"
val_tag = "scanrefer#scan2cap#scanqa#sqa3d#multi3dref"

evaluate = False
debug = False
distributed = True

if debug:
    enable_wandb = False
    gpu_num = 1
    do_save = False
    other_info = "debug"
else:
    enable_wandb = False
    gpu_num = 2
    do_save = True
    other_info = "descrip3d"

tag = f"{train_tag}__{val_tag}__{other_info}"

pretrained_path = "" # "./ckpts/ckpt_02.pth"

OUTPUT_DIR = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{lr}_ep{epoch}_{tag}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


node_rank = int(os.environ.get("SLURM_NODEID", 0))
nnodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
gpus_per_node = 2
world_size = nnodes * gpus_per_node

command = [
    "torchrun",
    "--nproc_per_node=2", 
    "tasks/train.py",
    f"{os.path.dirname(__file__)}/{config}config.py",
    "output_dir", OUTPUT_DIR,
    "scheduler.epochs", str(epoch),
    "optimizer.lr", str(lr),
    "model.add_scene_token", str(add_scene_token),
    "model.add_img_token", str(add_img_token),
    "pretrained_path", pretrained_path,
    "evaluate", str(evaluate),
    "wandb.enable", str(enable_wandb),
    "gpu_num", str(gpu_num),
    "distributed", str(distributed),
    "do_save", str(do_save),
    "batch_size", str(batch_size),
    "model.train_emb", str(train_emb),
    "model.train_img_proj", str(train_img_proj),
    "train_tag", train_tag,
    "val_tag", val_tag,
    "model.no_obj", str(no_obj),
    "model.input_dim", str(input_dim),
    "model.bidirection", str(bidirection),
    "optimizer.different_lr.enable", str(different_lr),
    "model.max_obj_num", str(max_obj_num),
    "lora.lora_r", str(lora_r),
    "lora.lora_alpha", str(lora_alpha),
    "model.add_pos_emb", str(add_pos_emb),
    "model.feat_fusion", str(feat_fusion),
    "optimizer.max_grad_norm", str(max_grad_norm),
    "seed", str(seed),
    "model.fuse_with_id", str(fuse_with_id),
    "model.llama_model_path", llama_model_path,
    "model.use_location_token", str(use_location_token),
]

print("Running command:", " ".join(command))
subprocess.run(command)
