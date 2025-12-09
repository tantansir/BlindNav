"""
Quick test script to verify YOLO + LLM integration works
"""

import sys
from pathlib import Path
from inference_yolo import YOLOInference
from llm_descriptor import LLMDescriptor

def test_yolo():
    """Test YOLO inference"""
    print("Testing YOLO inference...")
    
    model_path = "runs_blindroad/yv8n_merged_v1/weights/best.pt"
    data_yaml = "blind_merged.v1/data.yaml"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not Path(data_yaml).exists():
        print(f"❌ Data file not found: {data_yaml}")
        return False
    
    try:
        yolo = YOLOInference(model_path, data_yaml)
        print(f"✓ YOLO loaded successfully")
        print(f"  Classes: {len(yolo.class_names)}")
        print(f"  Sample classes: {yolo.class_names[:5]}...")
        return True
    except Exception as e:
        print(f"❌ YOLO error: {e}")
        return False

def test_llm():
    """Test LLM connection"""
    print("\nTesting LLM connection...")
    
    llm = LLMDescriptor(model_name="llama3.2")
    
    if llm.check_connection():
        print("✓ Ollama is running")
        
        if llm.check_model():
            print("✓ Model 'llama3.2' is available")
            
            # Test generation
            test_summary = "Detected objects: 1 red-light in the top-center area (confidence: 92%), 1 Crosswalk in the bottom-center area (confidence: 78%)"
            print("\nTesting description generation...")
            description = llm.generate_description(test_summary)
            print(f"✓ Generated description:\n  {description}")
            return True
        else:
            print("⚠ Model 'llama3.2' not found. Install with: ollama pull llama3.2")
            return False
    else:
        print("⚠ Ollama is not running. Start with: ollama serve")
        print("  (This is OK if you plan to use --no-llm mode)")
        return False

def test_simple_description():
    """Test simple description fallback"""
    print("\nTesting simple description (no LLM)...")
    
    llm = LLMDescriptor()
    test_summary = "Detected objects: 1 red-light in the top-center area (confidence: 92%), 2 cars in the center area (confidence: 85%)"
    description = llm.generate_simple_description(test_summary)
    print(f"✓ Simple description:\n  {description}")
    return True

if __name__ == "__main__":
    print("="*60)
    print("Blind Navigation Assistant - Integration Test")
    print("="*60)
    
    yolo_ok = test_yolo()
    llm_ok = test_llm()
    simple_ok = test_simple_description()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"YOLO: {'✓' if yolo_ok else '❌'}")
    print(f"LLM:  {'✓' if llm_ok else '⚠ (optional)'}")
    print(f"Simple: {'✓' if simple_ok else '❌'}")
    
    if yolo_ok:
        print("\n✅ System is ready! You can run:")
        print("   python blind_navigation_assistant.py --image <image_path>")
        if not llm_ok:
            print("   (Use --no-llm flag if you don't want to use LLM)")
    else:
        print("\n❌ Please fix YOLO issues before proceeding")
        sys.exit(1)

