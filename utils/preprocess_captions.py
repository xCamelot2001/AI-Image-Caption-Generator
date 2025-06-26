import os
import string
import json
import csv
from collections import defaultdict

def load_captions_csv(filepath):
    captions = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image'].strip()
            caption = row['caption'].strip()
            captions[image_id].append(caption)
    return captions

def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.strip()
    caption = ' '.join(caption.split())
    return caption

def clean_all_captions(captions_dict):
    cleaned = {}
    for img, caps in captions_dict.items():
        cleaned[img] = [clean_caption(c) for c in caps]
    return cleaned

def save_captions(captions_dict, out_path):
    with open(out_path, 'w') as f:
        json.dump(captions_dict, f, indent=2)

if __name__ == "__main__":
    raw_txt_path = 'data/flickr8k_text.txt'  # CSV format
    output_json_path = 'data/cleaned_captions.json'

    print("📥 Loading and cleaning captions (CSV format)...")
    raw_captions = load_captions_csv(raw_txt_path)
    clean_captions = clean_all_captions(raw_captions)
    save_captions(clean_captions, output_json_path)
    print(f"✅ Captions saved to: {output_json_path}")
