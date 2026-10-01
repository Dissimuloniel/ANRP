import os
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

NUM_CLASSES = 23
BATCH_SIZE = 8
VAL_SPLIT = 0.2

def get_vgg19():
    model = models.vgg19(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model

def main():
    #DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #TRANSFORMS
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    #DATASET
    dataset_path = "D:\\New_dataset"
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    print(f"Total images found: {len(full_dataset)}, Classes: {full_dataset.classes}")

    #SPLIT
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    #DATALOADERS
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=(device.type=='cuda'))
    #MODEL
    model = get_vgg19().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    #TRAINING
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        i = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            print("Итерация:", i)
            i += 1
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        #VALIDATION
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")
    #SAVE MODEL
    torch.save(model.state_dict(), "vgg19_chars.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()