"""
Example usage of the Blind Navigation Assistant
Shows how to use the components programmatically
"""

from inference_yolo import YOLOInference
from llm_descriptor import LLMDescriptor

def example_single_image():
    """Example: Process a single image"""
    print("="*60)
    print("Example 1: Process Single Image")
    print("="*60)
    
    # Initialize YOLO
    yolo = YOLOInference(
        model_path="runs_blindroad/yv8n_merged_v1/weights/best.pt",
        data_yaml="blind_merged.v1/data.yaml",
        device="0"  # Use GPU, or "cpu" for CPU
    )
    
    # Initialize LLM (optional)
    llm = LLMDescriptor(model_name="llama3.2")
    use_llm = llm.check_connection() and llm.check_model()
    
    if not use_llm:
        print("Note: LLM not available, using simple descriptions")
    
    # Process image (use a test image from your dataset)
    image_path = "blind_merged.v1/test/images"  # Update with actual image path
    print(f"\nProcessing: {image_path}")
    
    # Detect objects
    detections = yolo.detect(image_path, conf_threshold=0.25)
    print(f"\nFound {len(detections)} detections:")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2%} at {det['center']}")
    
    # Format for LLM
    detection_summary = yolo.format_detections_for_llm(detections)
    print(f"\nDetection Summary:\n{detection_summary}")
    
    # Generate description
    if use_llm:
        description = llm.generate_description(detection_summary)
    else:
        description = llm.generate_simple_description(detection_summary)
    
    print(f"\nDescription:\n{description}")


def example_batch_processing():
    """Example: Process multiple images"""
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    
    from pathlib import Path
    
    yolo = YOLOInference(
        model_path="runs_blindroad/yv8n_merged_v1/weights/best.pt",
        data_yaml="blind_merged.v1/data.yaml"
    )
    
    llm = LLMDescriptor()
    use_llm = llm.check_connection() and llm.check_model()
    
    # Process all test images
    test_dir = Path("blind_merged.v1/test/images")
    if test_dir.exists():
        image_files = list(test_dir.glob("*.jpg"))[:5]  # Process first 5
        
        for img_path in image_files:
            detections = yolo.detect(str(img_path), conf_threshold=0.25)
            summary = yolo.format_detections_for_llm(detections)
            
            if use_llm:
                desc = llm.generate_description(summary)
            else:
                desc = llm.generate_simple_description(summary)
            
            print(f"\n{img_path.name}:")
            print(f"  {desc}")


def example_custom_prompt():
    """Example: Custom LLM prompt"""
    print("\n" + "="*60)
    print("Example 3: Custom LLM Prompt")
    print("="*60)
    
    yolo = YOLOInference(
        model_path="runs_blindroad/yv8n_merged_v1/weights/best.pt",
        data_yaml="blind_merged.v1/data.yaml"
    )
    
    llm = LLMDescriptor(model_name="llama3.2")
    
    # Custom context for the LLM
    custom_context = """You are a helpful navigation assistant for blind people. 
    Focus on safety-critical information first (traffic lights, vehicles), 
    then navigation aids (crosswalks, tactile paths). Be very concise (1-2 sentences)."""
    
    # This would require modifying llm_descriptor.py to accept custom context
    # For now, the default context is used
    print("Note: Custom prompts can be added by modifying llm_descriptor.py")


if __name__ == "__main__":
    print("Blind Navigation Assistant - Usage Examples")
    print("\nNote: Update image paths in the code before running")
    
    # Uncomment to run examples:
    # example_single_image()
    # example_batch_processing()
    # example_custom_prompt()
    
    print("\n" + "="*60)
    print("To run examples, uncomment the function calls above")
    print("Or use the main script:")
    print("  python blind_navigation_assistant.py --image <path>")
    print("="*60)

