import torch
import segmentation_models_pytorch as smp

def test_model_architecture():
    # Verifies the U-Net model can be initialized
    model = smp.Unet(encoder_name="resnet34", classes=1)
    assert model is not None
    print("Model architecture verified!")