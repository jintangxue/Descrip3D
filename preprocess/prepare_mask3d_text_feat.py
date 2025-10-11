import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def generate_text_features(desc_json_path, output_path, encoder_name="all-MiniLM-L6-v2"):
    print(f"Loading descriptions from: {desc_json_path}")
    with open(desc_json_path, 'r') as f:
        desc_data = json.load(f)

    encoder = SentenceTransformer("all-mpnet-base-v2")
    encoder = encoder.to("cuda" if torch.cuda.is_available() else "cpu")

    text_features = {}

    print(f"Encoding {len(desc_data)} descriptions...")
    for item in tqdm(desc_data):
        scene_id = item["scene_id"]
        obj_id = item["object_id"]
        desc = item["object_description"]

        key = f"{scene_id}_{obj_id:02}"
        embedding = encoder.encode(desc, normalize_embeddings=True)
        text_features[key] = torch.tensor(embedding, dtype=torch.float)

    print(f"Saving to {output_path}")
    torch.save(text_features, output_path)
    print("Done.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate .pt file of text features from object descriptions")
    parser.add_argument("--desc_json", type=str, default="./object_descriptions_all_scenes_cleaned.json", help="Path to the JSON file with descriptions")
    parser.add_argument("--output", type=str, default="./object_descriptions_all_scenes_cleaned.pt", help="Path to save the generated .pt file")
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")

    args = parser.parse_args()

    generate_text_features(args.desc_json, args.output, args.encoder)

