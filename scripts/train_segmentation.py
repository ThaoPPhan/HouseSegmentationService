import os
import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A  # Standard library for CV augmentation
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt

# --- Phase 1: Dataset Class ---
class HouseDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.img_dir = os.path.join(split_dir, 'images')
        self.mask_dir = os.path.join(split_dir, 'masks')
        self.images = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype("float32") # Convert to binary 0/1 [cite: 120]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        return image, mask.unsqueeze(0)

# --- Phase 2: Augmentation (Week 7 Core Concepts) ---
# Prevents overfitting by exposing model to diverse data [cite: 69]
train_transform = A.Compose([
    A.Rotate(limit=30, p=0.5),      # capture invariance to object orientation [cite: 44, 45]
    A.HorizontalFlip(p=0.5),        # captures invariance to mirroring [cite: 43]
    A.RandomBrightnessContrast(p=0.2), # simulates lighting conditions [cite: 60]
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([A.Normalize(), ToTensorV2()])

# --- Phase 3: Training Setup ---
# Use Transfer Learning with U-Net and ResNet backbone [cite: 14, 124]
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = smp.losses.DiceLoss(mode='binary') # Dice score is robust [cite: 136]

# --- Phase 4: Data Loaders (Pointing to your folders) ---
train_ds = HouseDataset("data/train", transform=train_transform)
val_ds = HouseDataset("data/val", transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# --- Phase 5: Training Loop ---
train_losses, val_losses = [], []
for epoch in range(10): # Adjust epochs as needed
    model.train()
    # Logic to track metrics [cite: 128]
    # Calculate IoU and Dice using your calculate_metrics function
    # Append losses to train_losses and val_losses for plotting
    print(f"Epoch {epoch} complete.")

# --- Phase 6: Save and Plot ---
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.savefig('outputs/loss_curves.png') # Requirement for report
torch.save(model.state_dict(), "house_segmentation_model.pth")