from ultralytics import YOLO
import torch
import os

def train_optimized_blind_road():
    """
    Advanced training with data augmentation and optimization
    """
    print("=== Advanced Blind Road Segmentation Training ===")
    
    # Device configuration
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load model - try larger model for better performance
    model = YOLO('yolov8s-seg.pt')  # Upgrade from nano to small
    
    # Advanced training configuration
    training_config = {
        'data': 'data.yaml',
        'epochs': 150,              # More epochs for convergence
        'imgsz': 640,
        'batch': 8,
        'workers': 4,
        'device': device,
        'seed': 42,
        'patience': 30,             # More patience for slow convergence
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        
        # Advanced optimizer settings
        'optimizer': 'AdamW',
        'lr0': 0.001,              # Initial learning rate
        'lrf': 0.01,               # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Enhanced augmentation
        'augment': True,
        'hsv_h': 0.015,            # Hue augmentation
        'hsv_s': 0.7,              # Saturation augmentation  
        'hsv_v': 0.4,              # Value augmentation
        'degrees': 45.0,           # Rotation augmentation
        'translate': 0.2,          # Translation augmentation
        'scale': 0.5,              # Scale augmentation
        'shear': 0.0,
        'perspective': 0.0005,     # Perspective augmentation
        'flipud': 0.0,
        'fliplr': 0.5,             # Horizontal flip
        'mosaic': 1.0,             # Mosaic augmentation
        'mixup': 0.2,              # Mixup augmentation
        'copy_paste': 0.2,         # Copy-paste augmentation
        
        # Loss weights - adjusted for segmentation
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'mask_ratio': 4,           # Mask loss ratio
        
        # Advanced training tricks
        'close_mosaic': 10,        # Disable mosaic last 10 epochs
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,            # Cosine learning rate scheduler
        'label_smoothing': 0.1,    # Label smoothing for better generalization
        
        'verbose': True,
        'amp': False,              # Disable for CPU
    }
    
    print("Advanced training configuration with data augmentation")
    
    # Start training
    results = model.train(**training_config)
    
    # Enhanced validation
    print("\n=== Enhanced Validation ===")
    val_results = model.val(
        data='data.yaml',
        split='test',  # Explicitly use test set
        conf=0.001,    # Low confidence to see all detections
        iou=0.6
    )
    
    print(f"Enhanced validation results: {val_results}")
    
    return model

if __name__ == "__main__":
    trained_model = train_optimized_blind_road()