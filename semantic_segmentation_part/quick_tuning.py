from ultralytics import YOLO
import time

def quick_hyperparameter_test():
    """
    Quick test of 2-3 most promising configurations
    """
    print("🚀 Quick Hyperparameter Testing")
    print("Testing only the most promising combinations...")
    
    # Only test the most promising combinations
    test_configs = [
        {'name': 'AdamW_Default', 'optimizer': 'AdamW', 'lr0': 0.001, 'augment': 0.3},
        {'name': 'AdamW_HighLR', 'optimizer': 'AdamW', 'lr0': 0.01, 'augment': 0.3},
        {'name': 'SGD_Default', 'optimizer': 'SGD', 'lr0': 0.01, 'augment': 0.1},
    ]
    
    best_map = 0
    best_config = {}
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"Config: {config}")
        
        start_time = time.time()
        
        try:
            model = YOLO('yolov8s-seg.pt')
            
            results = model.train(
                data='data.yaml',
                epochs=30,  # Reduced epochs for quick testing
                imgsz=640,
                batch=8,
                device='cpu',
                lr0=config['lr0'],
                optimizer=config['optimizer'],
                hsv_h=config['augment'] * 0.015,
                hsv_s=config['augment'] * 0.7,
                hsv_v=config['augment'] * 0.4,
                save=False,
                verbose=False,
                patience=10,  # Early stopping
            )
            
            # Quick validation
            val_results = model.val(split='test', verbose=False)
            current_map = val_results.box.map50  # mAP50
            
            training_time = time.time() - start_time
            print(f"✅ mAP50: {current_map:.4f} | Time: {training_time/60:.1f} min")
            
            if current_map > best_map:
                best_map = current_map
                best_config = config
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    print(f"\n🎯 Best configuration: {best_config}")
    print(f"🎯 Best mAP50: {best_map:.4f}")
    
    return best_config

if __name__ == "__main__":
    best_config = quick_hyperparameter_test()