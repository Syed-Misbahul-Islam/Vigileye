import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/raw/Driver Drowsiness Dataset (DDD)")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "drowsiness_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("VIGILEYE DROWSINESS MODEL TRAINING")
print("=" * 60)
print(f"Dataset : {DATASET_DIR}")
print(f"Device  : {device}")
print(f"Batch   : {BATCH_SIZE}")
print(f"Epochs  : {EPOCHS}")
print("=" * 60)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_DIR.resolve()}"
    )

class_dirs = [
    p for p in DATASET_DIR.iterdir()
    if p.is_dir()
]

print("\nClasses found:")

for class_dir in class_dirs:
    image_count = sum(
        1
        for f in class_dir.rglob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"  {class_dir.name}: {image_count} images")


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=5),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

full_dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=train_transform
)

print("\nClass mapping:")
print(full_dataset.class_to_idx)

total_size = len(full_dataset)

val_size = int(total_size * VAL_RATIO)
test_size = int(total_size * TEST_RATIO)
train_size = total_size - val_size - test_size

print("\nDataset split:")
print(f"  Training   : {train_size}")
print(f"  Validation : {val_size}")
print(f"  Testing    : {test_size}")
print(f"  Total      : {total_size}")


# ============================================================
# SPLIT DATASET
# ============================================================

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

# Use evaluation transforms for validation/test.
val_dataset.dataset.transform = eval_transform
test_dataset.dataset.transform = eval_transform


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading MobileNetV3...")

weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)

# Replace final classifier.
in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    2
)

model = model.to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# TRAIN
# ============================================================

best_val_accuracy = 0.0

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": full_dataset.class_to_idx,
                "image_size": IMAGE_SIZE
            },
            MODEL_PATH
        )

        print(
            f"Saved best model → {MODEL_PATH}"
        )


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST")
print("=" * 60)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_accuracy = evaluate(
    model,
    test_loader
)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

print("\nClass mapping:")
print(checkpoint["class_to_idx"])

print("\nTraining complete.")
print(f"Model saved at: {MODEL_PATH}")