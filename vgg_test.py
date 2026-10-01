import torch
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image

# --- Настройки ---
NUM_CLASSES = 23
MODEL_PATH = "vgg19_chars_32.pth"
DATASET_PATH = "D:\\New_dataset"
IMAGE_PATH = "D:\\14.jpg"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_vgg19():
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    for param in model.features[:20].parameters():
        param.requires_grad = False
    model.classifier[6] = torch.nn.Sequential(
        torch.nn.Linear(4096, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, NUM_CLASSES)
    )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_vgg19()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_idx = predicted.item()

dataset = ImageFolder(DATASET_PATH)
class_name = dataset.classes[predicted_idx]
print(f"Predicted class index: {predicted_idx}")
print(f"Predicted class name: {class_name}")