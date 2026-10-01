import os
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# =========================
# Настройки
# =========================
NUM_CLASSES = 23
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42

EPOCHS_STAGE1 = 10   # обучаем только classifier
EPOCHS_STAGE2 = 10   # дообучаем classifier + последний conv-блок

PATIENCE = 5         # early stopping
LOG_FILE = "training_log.txt"
BEST_MODEL_PATH = "vgg19_chars_best.pth"

DATASET_PATH = "D:\\New_dataset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Фиксация random seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Модель
# =========================
def get_vgg19():
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

    # Замораживаем весь backbone на первом этапе
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    return model


# =========================
# Train / Val transforms
# =========================
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=4),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            shear=3
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


# =========================
# Разделение датасета
# =========================
def build_datasets(dataset_path, train_transform, val_transform, val_split=0.2, seed=42):
    # Два одинаковых ImageFolder с разными transform
    full_train_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(dataset_path, transform=val_transform)

    total_size = len(full_train_dataset)
    indices = list(range(total_size))

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    return train_dataset, val_dataset, full_train_dataset.classes


# =========================
# Обучение одной эпохи
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# =========================
# Валидация
# =========================
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# =========================
# Один этап обучения
# =========================
def run_stage(
    stage_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    patience,
    device,
    start_epoch_idx,
    best_acc,
    best_weights
):
    no_improve_epochs = 0
    history = []

    for epoch in range(epochs):
        global_epoch = start_epoch_idx + epoch + 1

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{stage_name}] Epoch {epoch + 1}/{epochs} "
            f"(Global {global_epoch}) | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        history.append({
            "stage": stage_name,
            "epoch": global_epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{stage_name},{global_epoch},{current_lr:.8f},"
                f"{train_loss:.4f},{train_acc:.2f},"
                f"{val_loss:.4f},{val_acc:.2f}\n"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, BEST_MODEL_PATH)
            print(f"  -> Новая лучшая модель сохранена: {BEST_MODEL_PATH} (Val Acc: {best_acc:.2f}%)")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  -> Без улучшения: {no_improve_epochs}/{patience}")

        if no_improve_epochs >= patience:
            print(f"Early stopping на этапе {stage_name}")
            break

    return best_acc, best_weights, history, len(history)


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Путь к датасету не найден: {DATASET_PATH}")

    train_transform, val_transform = get_transforms()

    train_dataset, val_dataset, class_names = build_datasets(
        DATASET_PATH,
        train_transform,
        val_transform,
        VAL_SPLIT,
        SEED
    )

    print(f"Всего изображений: {len(train_dataset) + len(val_dataset)}")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Классы ({len(class_names)}): {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = get_vgg19().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("stage,epoch,lr,train_loss,train_acc,val_loss,val_acc\n")

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    total_done_epochs = 0

    # =========================
    # ЭТАП 1: только classifier
    # =========================
    print("\n========== Stage 1: train classifier ==========")

    optimizer_stage1 = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler_stage1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage1,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_acc, best_weights, _, done_epochs = run_stage(
        stage_name="stage1",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        epochs=EPOCHS_STAGE1,
        patience=PATIENCE,
        device=DEVICE,
        start_epoch_idx=total_done_epochs,
        best_acc=best_acc,
        best_weights=best_weights
    )
    total_done_epochs += done_epochs

    # =========================
    # ЭТАП 2: разморозка последнего conv-блока
    # =========================
    print("\n========== Stage 2: unfreeze last conv block ==========")

    # У VGG19 последний conv-блок примерно начинается с features[28]
    for param in model.features[28:].parameters():
        param.requires_grad = True

    optimizer_stage2 = torch.optim.AdamW([
        {
            "params": model.features[28:].parameters(),
            "lr": 1e-5,
            "weight_decay": 1e-4
        },
        {
            "params": model.classifier.parameters(),
            "lr": 5e-5,
            "weight_decay": 1e-4
        }
    ])

    scheduler_stage2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage2,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_acc, best_weights, _, done_epochs = run_stage(
        stage_name="stage2",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        epochs=EPOCHS_STAGE2,
        patience=PATIENCE,
        device=DEVICE,
        start_epoch_idx=total_done_epochs,
        best_acc=best_acc,
        best_weights=best_weights
    )
    total_done_epochs += done_epochs

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), BEST_MODEL_PATH)

    print("\n========== Training finished ==========")
    print(f"Лучшая val accuracy: {best_acc:.2f}%")
    print(f"Лучшая модель сохранена в: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()