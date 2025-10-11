import logging
import os
import json
import re
import random
import torch
import numpy as np
from dataset.base_dataset import BaseDataset, update_caption
from prompts.prompts import obj_caption_wid_prompt
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

OBJ_TAG_RE = re.compile(r"<OBJ(\d{1,3})>")

def _shorten_desc(desc: str, max_words: int = 10) -> str:
    if not desc:
        return ""
    t = desc.strip()
    t = re.split(r"[.;]\s*", t)[0]
    t = re.sub(r"\b(is|are)\s+(located|situated|positioned)\b", "is", t, flags=re.I)
    t = re.sub(r"\b(which is|that is)\b", "", t, flags=re.I)
    t = re.sub(r"\b(in the room|in the background)\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]) + " …"
    return t.strip()

def _mentioned_ids(text: str):
    return [int(m.group(1)) for m in OBJ_TAG_RE.finditer(text)]

class TrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, ann_list, config, **kwargs):
        super().__init__()
        self.feat_dim = config.model.input_dim
        self.img_feat_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num

        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
        text_feat_file = ann_list[4] if len(ann_list) > 4 else None
        
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = json.load(open(anno_file, 'r'))
        
        desc_file = config.get("description_file", None)
        self.max_desc = config.get("max_prompt_descriptions", 1)
        self.description_map = {}
        
        if desc_file and os.path.exists(desc_file):
            with open(desc_file, "r") as f:
                data = json.load(f)
                for entry in data:
                    scene_id = entry["scene_id"]
                    obj_id = int(entry["object_id"])
                    name = entry["object_name"]
                    desc = entry["object_description"].strip()
                    if scene_id not in self.description_map:
                        self.description_map[scene_id] = {}
                    self.description_map[scene_id][obj_id] = (name, desc)

        if len(ann_list) > 5:
            sample_ratio = ann_list[-1]
            if sample_ratio < 1:
                self.anno = random.sample(self.anno, int(sample_ratio * len(self.anno)))
        
        if feat_file in TrainDataset.cached_feats and img_feat_file in TrainDataset.cached_feats:
            self.scene_feats, self.scene_masks = TrainDataset.cached_feats[feat_file]
            self.scene_img_feats = TrainDataset.cached_feats[img_feat_file]
        else:
            self.feats = torch.load(feat_file, map_location='cpu') if feat_file and os.path.exists(feat_file) else None
            self.img_feats = torch.load(img_feat_file, map_location='cpu') if img_feat_file and os.path.exists(img_feat_file) else None
            self.text_feats = torch.load(text_feat_file, map_location='cpu') if text_feat_file and os.path.exists(text_feat_file) else None
            if self.attributes is None:
                self.scene_feats = self.feats
                self.scene_img_feats = self.scene_masks = self.scene_text_feats = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks, self.scene_text_feats = self.prepare_scene_features()
            TrainDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            TrainDataset.cached_feats[img_feat_file] = self.scene_img_feats

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        if self.attributes is not None and self.anno[index]['scene_id'] not in self.attributes:
            return self.__getitem__(random.randint(0, len(self.anno)-1))
        obj_id = int(self.anno[index].get("obj_id", random.randint(0, self.max_obj_num - 1)))
        question = self.anno[index].get("prompt", random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>"))
        caption = self.anno[index]["caption"]
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids, scene_text_feat = self.get_anno(index)

        allowed_ids = set(assigned_ids.tolist())
        desc_map = self.description_map.get(scene_id, {})
        desc_list, used_ids = [], set()

        REL_RE = re.compile(r"\b(next to|near|behind|in front of|on top of|under|between|inside|left|right|closest|farthest)\b", re.I)
        should_inject = self.max_desc > 0 and ("<OBJ" in question or REL_RE.search(question))

        if should_inject:
            if len(desc_list) < self.max_desc:
                candidates = []
                q_lc = question.lower()
                for oid, (name, desc) in desc_map.items():
                    if oid in used_ids or oid not in allowed_ids:
                        continue
                    n = (name or "").strip().lower()
                    if not n:
                        continue
                    m = re.search(r"\b" + re.escape(n) + r"(?:s|es)?\b", q_lc)
                    if m:
                        candidates.append((m.start(), oid, desc))
                candidates.sort(key=lambda x: x[0])

                for _, oid, desc in candidates:
                    short = _shorten_desc(desc, max_words=10)
                    if short:
                        desc_list.append(f"<OBJ{oid:03}>: {short}")
                        used_ids.add(oid)
                        if len(desc_list) >= self.max_desc:
                            break

        if should_inject and desc_list:
            question += "\n[Generated description (may be noisy)] " + " ".join(desc_list)

        caption = update_caption(caption, assigned_ids)
        question = update_caption(question, assigned_ids)

        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, caption, question, scene_text_feat

def train_collate_fn(batch):
    (
        scene_feats, scene_img_feats, scene_masks, scene_locs,
        obj_ids, assigned_ids, captions, questions, scene_text_feats
    ) = zip(*batch)
    return {
        "scene_feat": pad_sequence(scene_feats, batch_first=True),
        "scene_img_feat": pad_sequence(scene_img_feats, batch_first=True),
        "scene_text_feat": pad_sequence(scene_text_feats, batch_first=True),
        "scene_locs": pad_sequence(scene_locs, batch_first=True),
        "scene_mask": pad_sequence(scene_masks, batch_first=True).to(torch.bool),
        "assigned_ids": pad_sequence(assigned_ids, batch_first=True),
        "obj_ids": torch.tensor(obj_ids),
        "answers": captions,
        "questions": questions
    }