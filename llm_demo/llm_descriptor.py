"""
LLM Integration Module
Connects to local LLM (Ollama) to generate natural language descriptions
"""

import requests
import json
from typing import Optional, Dict
import time


class LLMDescriptor:
    """Interface to local LLM for generating descriptions"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2",
                 timeout: int = 30):
        """
        Initialize LLM descriptor
        
        Args:
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            model_name: Name of the model to use (default: llama3.2)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"
        
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_model(self) -> bool:
        """Check if the specified model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                return any(self.model_name in name for name in model_names)
            return False
        except:
            return False
    
    def generate_description(self, 
                            detection_summary: str,
                            context: str = "You are a helpful assistant describing the environment to a blind person. Be concise, clear, and focus on navigation-relevant information.") -> str:
        """
        Generate natural language description from YOLO detections
        
        Args:
            detection_summary: Formatted string of YOLO detections
            context: System context/prompt for the LLM
            
        Returns:
            Natural language description
        """
        prompt = f"""You are a safety-first mobility assistant for a blind person.
Based on the detections, speak ONE short sentence (<= 18 words), imperative voice.
Order of priority: 1) stop/wait warnings, 2) crossing guidance, 3) nearest hazard direction.

{detection_summary}

Output rules:
- One sentence only, <= 18 words.
- Start with an imperative verb (e.g., "Stop", "Wait", "Cross", "Keep left", "Proceed carefully").
- Mention direction (left/center/right) if relevant.
- Do NOT explain confidence or analysis.

Example outputs:
- "Stop; red light ahead, cars center; wait at curb."
- "Green light ahead, cross at center crosswalk now."
- "Keep right; car ahead center; proceed carefully."

Now speak the one-sentence guidance:"""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 60
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Unable to generate description.').strip()
            else:
                return f"Error: LLM API returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running and the model is installed."
        except requests.exceptions.Timeout:
            return "Error: LLM request timed out."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_simple_description(self, detection_summary: str) -> str:
        """
        Generate a simpler description without LLM (fallback)
        Converts detection summary into more natural language
        """
        if "No objects detected" in detection_summary:
            return "I don't detect any objects in front of you. Please proceed with caution."
        
        # Extract key information
        lines = detection_summary.replace("Detected objects: ", "").replace(".", "").split(", ")
        
        # Prioritize important objects for navigation
        priority_objects = {
            'traffic-light': 'traffic light',
            'red-light': 'red traffic light',
            'green-light': 'green traffic light',
            'yellow-light': 'yellow traffic light',
            'Crosswalk': 'crosswalk',
            'Walk': 'walk signal',
            'Don-t Walk': 'don\'t walk signal',
            'tactile-guide-path': 'tactile guide path',
            'car': 'car',
            'bus': 'bus',
            'truck': 'truck',
            'bicycle': 'bicycle',
            'motorbike': 'motorcycle',
            'person': 'person'
        }
        
        important = []
        other = []
        
        for line in lines:
            found_important = False
            for key, desc in priority_objects.items():
                if key.lower() in line.lower():
                    important.append(line.replace(key, desc))
                    found_important = True
                    break
            if not found_important:
                other.append(line)
        
        description_parts = []
        
        if important:
            description_parts.append("Important for navigation: " + ", ".join(important))
        if other:
            description_parts.append("Also visible: " + ", ".join(other))
        
        return ". ".join(description_parts) + "."

