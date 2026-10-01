import os
import torch
from torchvision import transforms
from PIL import Image

input_path = "D:\\New_dataset\\A\\0.jpg"
output_dir = "D:\\New_dataset_New\\A"
os.makedirs(output_dir, exist_ok=True)

image = Image.open(input_path).convert("RGB")

def add_noise(x):
    if torch.rand(1).item() < 0.5:
        noise = 0.03 * torch.randn_like(x)
        x = torch.clamp(x + noise, 0.0, 1.0)
    return x

transform = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomPerspective(distortion_scale=0.15, p=1.0)
    ], p=0.3),

    transforms.RandomRotation(degrees=5),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        shear=3
    ),

    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.25, contrast=0.25)
    ], p=0.5),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
    ], p=0.3),

    transforms.ToTensor(),
    transforms.Lambda(add_noise),
    transforms.ToPILImage()
])

for i in range(30):
    aug = transform(image)
    aug.save(os.path.join(output_dir, f"aug_{i}.jpg"))

print("Готово!")