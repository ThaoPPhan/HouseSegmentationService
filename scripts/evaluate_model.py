import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from train_segmentation import HouseDataset, val_transform # Reuse your logic

# 1. Load Model
model = smp.Unet(encoder_name="resnet34", classes=1)
model.load_state_dict(torch.load("house_segmentation_model.pth"))
model.eval()

# 2. Setup Test Data
test_ds = HouseDataset("data/test", transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=1)

# 3. Calculate Metrics
total_iou = 0
total_dice = 0

with torch.no_grad():
    for images, masks in test_loader:
        outputs = model(images)
        # Apply sigmoid to get probabilities between 0 and 1
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        # Intersection over Union (IoU) calculation [cite: 130]
        tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        
        total_iou += iou
        total_dice += dice

print(f"--- Final Test Metrics ---")
print(f"Mean IoU: {total_iou / len(test_loader):.4f}")
print(f"Mean Dice Score: {total_dice / len(test_loader):.4f}")