# Blind Navigation Assistant - YOLO + LLM Integration

This project combines YOLO object detection with a local LLM to provide natural language descriptions of street scenes for blind and visually impaired users.

## Overview

The system:
1. **Detects objects** using a trained YOLO model (20 classes including traffic lights, crosswalks, vehicles, tactile paths)
2. **Formats detections** into structured information
3. **Generates descriptions** using a local LLM (Ollama) to create natural, navigation-focused descriptions

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Setup Ollama (for LLM)

#### Install Ollama:
- **macOS**: `brew install ollama` or download from https://ollama.ai
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download from https://ollama.ai

#### Start Ollama:
```bash
ollama serve
```

#### Pull a model (in a new terminal):
```bash
# Recommended: llama3.2 (smaller, faster)
ollama pull llama3.2

# Or use other models:
ollama pull llama3.1
ollama pull mistral
ollama pull phi3
```

### 3. Verify Model Files

Make sure these files exist:
- `runs_blindroad/yv8n_merged_v1/weights/best.pt` (trained YOLO model)
- `blind_merged.v1/data.yaml` (class definitions)

## Usage

### Process a Single Image

```bash
python blind_navigation_assistant.py --image path/to/image.jpg
```

### Use Camera Feed

```bash
python blind_navigation_assistant.py --camera 0
```

Press:
- `q` to quit
- `s` to print/speak current description

### Options

```bash
python blind_navigation_assistant.py \
    --image test_image.jpg \
    --conf 0.3 \
    --llm-model llama3.2 \
    --device 0
```

**Arguments:**
- `--model`: Path to YOLO model (default: `runs_blindroad/yv8n_merged_v1/weights/best.pt`)
- `--data`: Path to data.yaml (default: `blind_merged.v1/data.yaml`)
- `--image`: Path to input image (if not provided, uses camera)
- `--camera`: Camera device ID (default: 0)
- `--conf`: Confidence threshold (default: 0.25)
- `--no-llm`: Disable LLM, use simple descriptions
- `--llm-model`: LLM model name (default: llama3.2)
- `--device`: Device for YOLO ("0" for GPU, "cpu" for CPU)

## Example Output

### Detection Summary:
```
Detected objects: 2 cars in the center area (confidence: 85%), 1 red-light in the top-center area (confidence: 92%), 1 Crosswalk in the bottom-center area (confidence: 78%)
```

### LLM Description:
```
I can see a red traffic light ahead, which means you should stop. There are 2 cars in the center of the scene. 
A crosswalk is visible in the lower portion, which you can use to cross safely when the light turns green.
```

### Simple Description (without LLM):
```
Important for navigation: 1 red traffic light, 1 crosswalk. Also visible: 2 cars.
```

## Architecture

```
┌─────────────┐
│   Image     │
│  (Camera)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   YOLO      │
│  Detection  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Detection  │────▶│     LLM     │
│  Formatter  │     │  (Ollama)   │
└─────────────┘     └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Description │
                    │  (Text/TTS) │
                    └─────────────┘
```

## Components

### 1. `inference_yolo.py`
- Loads trained YOLO model
- Performs object detection
- Formats detections with position information

### 2. `llm_descriptor.py`
- Connects to Ollama API
- Generates natural language descriptions
- Falls back to simple descriptions if LLM unavailable

### 3. `blind_navigation_assistant.py`
- Main application
- Combines YOLO + LLM
- Supports image files and camera feed

## Troubleshooting

### LLM Connection Issues

**Error: "Could not connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Check if it's accessible: `curl http://localhost:11434/api/tags`

**Error: "Model not found"**
- Install the model: `ollama pull llama3.2`
- Or use `--no-llm` to use simple descriptions

### YOLO Issues

**Error: "Model file not found"**
- Check that `runs_blindroad/yv8n_merged_v1/weights/best.pt` exists
- Or specify custom path with `--model`

**Slow inference:**
- Use GPU: `--device 0`
- Or use CPU: `--device cpu`
- Reduce image size or increase confidence threshold

## Future Enhancements

- [ ] Text-to-Speech (TTS) integration for audio output
- [ ] Real-time audio streaming descriptions
- [ ] Distance estimation for detected objects
- [ ] Direction guidance ("turn left", "straight ahead")
- [ ] Integration with navigation apps
- [ ] Support for multiple LLM backends (not just Ollama)
- [ ] Confidence-based filtering and prioritization
- [ ] Historical context (remember previous detections)

## Testing

Test with sample images:
```bash
python blind_navigation_assistant.py --image blind_merged.v1/test/images/[any_test_image].jpg
```

## Notes

- The system works without LLM (using `--no-llm`) but descriptions are simpler
- For production use, consider adding TTS for hands-free operation
- Camera feed processes every 5th frame for performance (adjustable in code)
- Confidence threshold of 0.25 is a good balance between accuracy and false positives

