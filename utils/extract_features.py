import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load pre-trained ResNet50 and remove last layer
def get_resnet_encoder():
    model = resnet50(pretrained=True)
    modules = list(model.children())[:-1]  # Remove FC layer
    model = torch.nn.Sequential(*modules)
    model.to(device)
    model.eval()
    return model

# Extract features from one image
def extract_feature(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu()

# Main function to loop over images
def extract_all_features(image_dir, output_path):
    model = get_resnet_encoder()
    features_dict = {}
    image_files = os.listdir(image_dir)
    for img_name in tqdm(image_files, desc="Extracting image features"):
        img_path = os.path.join(image_dir, img_name)
        try:
            features = extract_feature(img_path, model)
            features_dict[img_name] = features.numpy().tolist()
        except Exception as e:
            print(f"Failed on {img_name}: {e}")
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(features_dict, f)
    print(f"✅ Saved features to: {output_path}")

if __name__ == "__main__":
    image_dir = 'data/flickr8k_dataset'
    output_path = 'data/image_features.json'
    extract_all_features(image_dir, output_path)
