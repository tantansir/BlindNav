"""
YOLO Inference Module
Loads trained YOLO model and performs object detection on images
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class YOLOInference:
    """Wrapper for YOLO model inference"""
    
    def __init__(self, model_path: str, data_yaml: str = None, device: str = "0"):
        """
        Initialize YOLO inference
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            data_yaml: Path to data.yaml (optional, for class names)
            device: Device to run on ("0" for GPU, "cpu" for CPU)
        """
        self.model = YOLO(model_path)
        self.device = device
        self.class_names = []
        
        # Load class names from data.yaml if provided
        if data_yaml and Path(data_yaml).exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', [])
        else:
            # Try to get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
    
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Perform object detection on an image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection dictionaries with keys:
            - class_id: Class index
            - class_name: Class name
            - confidence: Detection confidence
            - bbox: Bounding box [x1, y1, x2, y2]
            - center: Center point [x, y]
        """
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                    
                    # Calculate center point
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': box.tolist(),
                        'center': [center_x, center_y]
                    })
        
        return detections
    
    def format_detections_for_llm(self, detections: List[Dict], image_size: Tuple[int, int] = None) -> str:
        """
        Format detections into a structured text description for LLM
        
        Args:
            detections: List of detection dictionaries
            image_size: Tuple of (width, height) for relative positioning
            
        Returns:
            Formatted string describing detections
        """
        if not detections:
            return "No objects detected in the scene."
        
        # Group detections by class
        class_counts = {}
        for det in detections:
            cls_name = det['class_name']
            if cls_name not in class_counts:
                class_counts[cls_name] = []
            class_counts[cls_name].append(det)
        
        # Build description
        description_parts = []
        
        for cls_name, dets in sorted(class_counts.items()):
            count = len(dets)
            avg_conf = np.mean([d['confidence'] for d in dets])
            
            if count == 1:
                det = dets[0]
                # Add position information if image size is provided
                if image_size:
                    width, height = image_size
                    rel_x = det['center'][0] / width
                    rel_y = det['center'][1] / height
                    
                    # Determine position
                    if rel_x < 0.33:
                        x_pos = "left"
                    elif rel_x > 0.67:
                        x_pos = "right"
                    else:
                        x_pos = "center"
                    
                    if rel_y < 0.33:
                        y_pos = "top"
                    elif rel_y > 0.67:
                        y_pos = "bottom"
                    else:
                        y_pos = "middle"
                    
                    position = f" in the {y_pos}-{x_pos} area"
                else:
                    position = ""
                
                description_parts.append(
                    f"1 {cls_name}{position} (confidence: {avg_conf:.1%})"
                )
            else:
                description_parts.append(
                    f"{count} {cls_name}s (average confidence: {avg_conf:.1%})"
                )
        
        return "Detected objects: " + ", ".join(description_parts) + "."

