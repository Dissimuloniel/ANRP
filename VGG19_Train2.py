import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

NUM_CLASSES = 22

def get_vgg19():
    model = models.vgg19(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model

def main():
    device = "cpu"
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder("D:\\dataset_sym", transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    model = get_vgg19().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 10
    for epoch in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch:", epoch, "Loss:", loss.item())
    torch.save(model.state_dict(), "models/vgg19_chars_2.pth")

if __name__ == "__main__":
    main()