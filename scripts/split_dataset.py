import os
import shutil
from sklearn.model_selection import train_test_split

# 1. Define Paths
RAW_IMG_DIR = 'data/raw_images'
RAW_MASK_DIR = 'data/raw_masks'
OUTPUT_DIR = 'data'

def create_folders():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'masks'), exist_ok=True)

def split_data():
    # Get all filenames (assuming image and mask have the same name)
    images = sorted([f for f in os.listdir(RAW_IMG_DIR) if f.endswith(('.jpg', '.png'))])
    
    # First split: Train (80%) and Temp (20%) 
    train_imgs, temp_imgs = train_test_split(images, test_size=0.20, random_state=42)
    
    # Second split: Split Temp into Val (10%) and Test (10%)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    split_map = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }

    for split, file_list in split_map.items():
        print(f"Moving {len(file_list)} images to {split}...")
        for filename in file_list:
            # Move Image
            shutil.copy(os.path.join(RAW_IMG_DIR, filename), 
                        os.path.join(OUTPUT_DIR, split, 'images', filename))
            # Move corresponding Mask (assuming same filename)
            shutil.copy(os.path.join(RAW_MASK_DIR, filename), 
                        os.path.join(OUTPUT_DIR, split, 'masks', filename))

if __name__ == "__main__":
    create_folders()
    split_data()
    print("Dataset split successfully!")