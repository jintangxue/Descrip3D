import json
import re
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./object_descriptions_all_scenes.json")
    ap.add_argument("--output", default="./object_descriptions_all_scenes_cleaned.json")
    args = ap.parse_args()

    input_file = args.input
    output_file = args.output

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        description = entry.get('object_description', '')
        
        description = re.sub(r'^\s*In the image,?\s*', '', description, flags=re.IGNORECASE)

        description = re.sub(r'^\s*In the scene,?\s*', '', description, flags=re.IGNORECASE)
        
        description = re.sub(r'\bwith a green label on it,?\s*', '', description, flags=re.IGNORECASE)

        description = re.sub(r'\bwith a green glowing text on it,?\s*', '', description, flags=re.IGNORECASE)

        description = re.sub(r'\bThe green glowing text on the wall might be related to the room\'s design or function, but the specific content of the text is not clear,?\s*', '', description, flags=re.IGNORECASE)

        description = re.sub(r'\bwith a green display of text on it,?\s*', '', description, flags=re.IGNORECASE)

        entry['object_description'] = description.strip()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Cleaned descriptions saved to: {output_file}")
