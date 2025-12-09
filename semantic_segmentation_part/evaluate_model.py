from ultralytics import YOLO

def evaluate_model():
    # Load the trained model
    model = YOLO('runs/segment/train/weights/best.pt')
    
    # Evaluate the model on the test set
    results = model.val(data='data.yaml', split='test')
    
    print("Test set evaluation results:")
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")
    
    # For segmentation, we can also look at mask metrics
    print(f"Mask mAP50: {results.seg.map50}")
    print(f"Mask mAP50-95: {results.seg.map}")

if __name__ == '__main__':
    evaluate_model()