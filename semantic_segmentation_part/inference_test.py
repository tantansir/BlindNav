from ultralytics import YOLO
import cv2
import os

def simple_inference_test():
    """
    Simple test to verify the trained model works
    """
    model_path = 'runs/segment/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("Loading trained model...")
    model = YOLO(model_path)
    
    # Test on a single image
    test_images = [
        'dataset/test/images/20_jpg.rf.589bdc34a7e8865fede536a6550e343e.jpg',
        'dataset/test/images/28_jpg.rf.334f2753906ce70f020261ed07eafbea.jpg'
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Test image not found: {image_path}")
            continue
            
        print(f"\nTesting on: {os.path.basename(image_path)}")
        
        # Run prediction
        results = model.predict(
            source=image_path,
            conf=0.25,      # Confidence threshold
            imgsz=640,
            save=False
        )
        
        result = results[0]
        
        # Print results
        if result.boxes is not None:
            num_detections = len(result.boxes)
            print(f"Detections: {num_detections}")
            
            # Show confidence scores
            boxes = result.boxes.data.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf, cls = box
                print(f"  Detection {i}: confidence = {conf:.3f}")
        else:
            print("No detections")
            
        if result.masks is not None:
            print(f"Segmentation masks: {len(result.masks)}")
        else:
            print("No segmentation masks")

if __name__ == "__main__":
    simple_inference_test()

def inference_with_tta(model, image_path):
    """
    Inference with Test Time Augmentation for better accuracy
    """
    # Regular inference
    results = model.predict(image_path, conf=0.25, imgsz=640)
    
    # TTA inference - averages predictions across augmentations
    tta_results = model.predict(image_path, conf=0.25, imgsz=640, augment=True)
    
    print(f"Regular detections: {len(results[0].boxes) if results[0].boxes else 0}")
    print(f"TTA detections: {len(tta_results[0].boxes) if tta_results[0].boxes else 0}")
    
    return tta_results[0]  # Return TTA results for better accuracy