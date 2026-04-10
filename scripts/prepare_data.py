import numpy as np
import cv2
import os

def generate_binary_mask(image_path, annotation_data):
    """
    Converts building coordinates/labels into a 2D pixel mask.
    Pixels belonging to a house = 1, Background = 0.
    """
    # Load image to get dimensions
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Example: Draw polygons from your dataset metadata onto the mask
    # cv2.fillPoly(mask, [coords], 1)
    
    return mask

def augment_data(image, mask):
    """
    Apply real-time augmentation to increase dataset diversity[cite: 24, 26].
    """
    # Flip horizontally (better performance sometimes than vertical) 
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        
    # Random Rotation (e.g., -30 to +30 degrees) [cite: 45]
    # This captures invariance to object orientation [cite: 44]
    
    return image, mask