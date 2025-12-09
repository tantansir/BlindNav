"""
Blind Navigation Assistant
Main application that combines YOLO object detection with LLM descriptions
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import cv2

from inference_yolo import YOLOInference
from llm_descriptor import LLMDescriptor


class BlindNavigationAssistant:
    """Main application for blind navigation assistance"""
    
    def __init__(self,
                 model_path: str,
                 data_yaml: str,
                 use_llm: bool = True,
                 llm_model: str = "llama3.2",
                 device: str = "0"):
        """
        Initialize the navigation assistant
        
        Args:
            model_path: Path to trained YOLO model
            data_yaml: Path to data.yaml with class definitions
            use_llm: Whether to use LLM for descriptions (False = simple fallback)
            llm_model: Name of LLM model to use
            device: Device for YOLO inference ("0" for GPU, "cpu" for CPU)
        """
        print("Initializing Blind Navigation Assistant...")
        
        # Initialize YOLO
        print(f"Loading YOLO model from {model_path}...")
        self.yolo = YOLOInference(model_path, data_yaml, device)
        print(f"✓ YOLO model loaded. {len(self.yolo.class_names)} classes available.")
        
        # Initialize LLM
        self.use_llm = use_llm
        if use_llm:
            print(f"Initializing LLM ({llm_model})...")
            self.llm = LLMDescriptor(model_name=llm_model)
            
            # Check connection
            if not self.llm.check_connection():
                print("⚠ Warning: Could not connect to Ollama. Falling back to simple descriptions.")
                print("  To use LLM, make sure Ollama is running: ollama serve")
                print("  And the model is installed: ollama pull llama3.2")
                self.use_llm = False
            elif not self.llm.check_model():
                print(f"⚠ Warning: Model '{llm_model}' not found. Falling back to simple descriptions.")
                print(f"  Install the model with: ollama pull {llm_model}")
                self.use_llm = False
            else:
                print(f"✓ LLM ready ({llm_model})")
        else:
            self.llm = LLMDescriptor(model_name=llm_model)
            print("✓ Using simple description mode (no LLM)")
    
    def process_image(self, image_path: str, conf_threshold: float = 0.25) -> dict:
        """
        Process an image and generate description
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with:
            - detections: List of raw detections
            - detection_summary: Formatted detection string
            - description: Natural language description
        """
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            return {
                'error': f'Could not load image: {image_path}',
                'detections': [],
                'detection_summary': '',
                'description': ''
            }
        
        height, width = img.shape[:2]
        
        # Run YOLO detection
        detections = self.yolo.detect(image_path, conf_threshold)
        
        # Format detections
        detection_summary = self.yolo.format_detections_for_llm(
            detections, 
            image_size=(width, height)
        )
        
        # Generate description
        if self.use_llm:
            description = self.llm.generate_description(detection_summary)
        else:
            description = self.llm.generate_simple_description(detection_summary)
        
        return {
            'detections': detections,
            'detection_summary': detection_summary,
            'description': description,
            'num_detections': len(detections)
        }
    
    def process_camera(self, camera_id: int = 0, conf_threshold: float = 0.25):
        """
        Process live camera feed
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
            conf_threshold: Confidence threshold for detections
        """
        print(f"\nStarting camera feed (Press 'q' to quit, 's' to speak description)...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        frame_count = 0
        process_every_n = 5  # Process every 5th frame for performance
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % process_every_n == 0:
                    # Save frame temporarily
                    temp_path = "temp_frame.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # Process
                    result = self.process_image(temp_path, conf_threshold)
                    
                    # Display on frame
                    if result['num_detections'] > 0:
                        # Draw detections
                        for det in result['detections']:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            cls_name = det['class_name']
                            conf = det['confidence']
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{cls_name} {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Show description on frame
                        desc_lines = result['description'].split('. ')
                        y_offset = 30
                        for line in desc_lines[:3]:  # Show first 3 lines
                            cv2.putText(frame, line[:50], (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            y_offset += 25
                    
                    # Print description
                    print(f"\n[{frame_count}] {result['description']}")
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Display frame
                cv2.imshow('Blind Navigation Assistant', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Speak description (if TTS is available)
                    print(f"\n🔊 {result.get('description', 'No description available')}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Blind Navigation Assistant - YOLO + LLM for accessibility'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='runs_blindroad/yv8n_merged_v1/weights/best.pt',
        help='Path to trained YOLO model'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='blind_merged.v1/data.yaml',
        help='Path to data.yaml'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image (if not provided, uses camera)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM and use simple descriptions'
    )
    
    parser.add_argument(
        '--llm-model',
        type=str,
        default='llama3.2',
        help='LLM model name (default: llama3.2)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device for YOLO (0 for GPU, cpu for CPU)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Initialize assistant
    assistant = BlindNavigationAssistant(
        model_path=args.model,
        data_yaml=args.data,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
        device=args.device
    )
    
    # Process image or camera
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        
        print(f"\nProcessing image: {args.image}")
        result = assistant.process_image(args.image, args.conf)
        
        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)
        print(f"\nDetections: {result['num_detections']}")
        print(f"\nDetection Summary:\n{result['detection_summary']}")
        print(f"\nDescription:\n{result['description']}")
        print("="*60)
    else:
        # Use camera
        assistant.process_camera(args.camera, args.conf)


if __name__ == "__main__":
    main()

