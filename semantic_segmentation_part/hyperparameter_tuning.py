from ultralytics import YOLO
import itertools

def hyperparameter_search():
    """
    Grid search for optimal hyperparameters
    """
    # Define parameter ranges
    learning_rates = [0.01, 0.001, 0.0001]
    optimizers = ['SGD', 'Adam', 'AdamW']
    augment_strengths = [0.1, 0.3, 0.5]
    
    best_map = 0
    best_params = {}
    
    for lr, optim, aug in itertools.product(learning_rates, optimizers, augment_strengths):
        print(f"\nTesting: lr={lr}, optimizer={optim}, aug={aug}")
        
        try:
            model = YOLO('yolov8s-seg.pt')
            
            results = model.train(
                data='data.yaml',
                epochs=50,  # Shorter for tuning
                imgsz=640,
                batch=4,
                device='cpu',
                lr0=lr,
                optimizer=optim,
                hsv_h=aug * 0.015,
                hsv_s=aug * 0.7,
                hsv_v=aug * 0.4,
                degrees=aug * 45.0,
                save=False,
                verbose=False
            )
            
            # Quick validation
            val_results = model.val(split='test', verbose=False)
            current_map = val_results.box.map50  # mAP50
            
            print(f"mAP50: {current_map:.4f}")
            
            if current_map > best_map:
                best_map = current_map
                best_params = {'lr': lr, 'optimizer': optim, 'augmentation': aug}
                
        except Exception as e:
            print(f"Failed with error: {e}")
            continue
    
    print(f"\n🎯 Best parameters: {best_params}")
    print(f"🎯 Best mAP50: {best_map:.4f}")
    
    return best_params

if __name__ == "__main__":
    best_params = hyperparameter_search()