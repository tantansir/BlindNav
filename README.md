# Blind Navigation System

- training dataset: https://drive.google.com/file/d/1Isqc7mHGZR8vmFGpl7AzRizI4IINzica/view
- implementation code: https://github.com/tantansir/BlindNav

A prototype system for navigation for visually impaired individuals, based on YOLO semantic segmentation and object detection.  
Currently, two modes are supported:

- `run.py`: Basic navigation version (detects sidewalks, zebra crossings, obstacles, and traffic lights)
- `run_llm.py`: Version integrated with LLM for generating scene descriptions via natural language (requires Ollama)

## Directory Structure

- `video/`                Example videos (e.g., `china1.mp4`, `pitts1.mp4`)
- `model/`                YOLO weights (semantic segmentation, object detection, traffic lights)
- `semantic_segmentation_part/`  Semantic segmentation related code
- `object_detection_part/`       Object detection related code
- `llm_demo/`             LLM experiment

### Object Detection Part

- **blind_merged.v2/**: Contains data configurations for object detection
  - `test/`, `train/`, `valid/`: Dataset folders (empty; all data are stored on Google Drive)
  - `blind_merge_config.json`: Configuration for merging the dataset
  - `data.yaml`: YAML file for dataset configuration
  - `merge_details.json`: Details for merging datasets
  - `sampling.py`: Sampling script for data processing

- **eda.py**: Exploratory data analysis script
- **merge_yolo_datasets.py**: Script for merging YOLO datasets
- **train_yolo.py**: Training script for YOLO object detection model
- **run.py**: Entry point for running object detection with YOLO

### Semantic Segmentation Part

- **eda_analysis.py**: Exploratory data analysis for segmentation datasets
- **evaluate_model.py**: Script to evaluate segmentation model performance
- **final_config.yaml**: Final configuration YAML for semantic segmentation
- **hyperparameter_tuning.py**: Hyperparameter tuning script for segmentation models
- **inference_test.py**: Inference testing script for segmentation models
- **quick_tuning.py**: Quick hyperparameter tuning script
- **train_seg_advanced.py**: Advanced training script for semantic segmentation models
- **verify_dataset.py**: Dataset verification script for segmentation models

---

## Environment Requirements

- Python 3.9 or higher (recommended)
- FFmpeg installed (for video decoding/encoding, required on some systems)
- GPU (optional, recommended for faster processing with Torch + CUDA)

### Core Dependencies

- `ultralytics` (YOLO model)
- `opencv-python` (cv2, for video reading/writing and drawing)
- `numpy`
- `pyttsx3` (optional, for text-to-speech)
- `requests` (only used by `run_llm.py` for Ollama integration)

Full dependencies are available in `requirements.txt`.

---

## Installation

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Model Weights

By default, the scripts will load weights from the `model/` directory:

- `model/yolo-seg.pt`           Semantic segmentation for blind path / zebra crossing
- `model/yolov8n.pt`            Object detection for pedestrians, vehicles, and obstacles
- `model/trafficlight_best.pt`  Traffic light detection for pedestrians

If the weight filenames differ, you can specify them via command-line arguments:

```bash
python run.py   --seg-weights path/to/seg.pt   --det-weights path/to/det.pt   --tl-weights path/to/trafficlight.pt
```

---

## Running Instructions

### 1. Run the basic version with example video

```bash
python run.py --source video/china1.mp4
```

Common arguments:

- `--source`: Video file path or camera ID (e.g., `0` for webcam)
- `--seg-weights`: Path to semantic segmentation weights
- `--det-weights`: Path to object detection weights
- `--tl-weights`: Path to traffic light detection weights (leave empty for no detection)
- `--save-path`: Output path for processed video (default: `output.mp4`)
- `--no-voice`: Disable voice announcements
- `--config`: Optional, specify a custom navigation configuration JSON

Example:

```bash
# Run with an example video and save the output
python run.py --source video/china1.mp4 --save-path output_china1.mp4

# Use webcam 0
python run.py --source 0
```

### 2. Run with LLM integration

`run_llm.py` builds on the basic navigation version, generating a brief natural language description of the current scene via the Ollama LLM.

```bash
python run_llm.py --source video/pitts1.mp4
```

Additional arguments:

- `--llm-model`: Ollama model name, default is `llama3.2`

Example:

```bash
python run_llm.py --source video/pitts1.mp4 --llm-model llama3.2
```

#### Ollama Configuration

1. Install Ollama  
   Follow the installation instructions for your system on the Ollama website.

2. Start the Ollama service

```bash
ollama serve
```

3. Pull a model (e.g., `llama3.2`)

```bash
ollama pull llama3.2
```

When starting the script, it will automatically detect:

- If LLM is available, it will display: `Press 'X' for scene description`
- If LLM is not available, it will prompt you on how to start or pull the model

---

## Interaction and Hotkeys

The window will display hotkey instructions, including:

- `Q` or `Esc`: Exit the program
- `Space` or `P`: Pause/Resume video playback
- `C`: Toggle "crosswalk mode" (detects zebra crossings and traffic lights)
- `X`: In `run_llm.py`, trigger LLM to generate a scene description and read it aloud (if voice is enabled)

---

## Custom Configuration

You can pass a JSON configuration file via `--config` to override some parameters in `NavigationConfig`, such as:

- Sidewalk / zebra crossing area threshold
- Obstacle area threshold
- Voice announcements settings and rate

Example (pseudo code):

```json
{
  "blind_path_ratio_threshold": 0.005,
  "crossing_ratio_threshold": 0.002,
  "voice_enabled": true,
  "voice_rate": 150
}
```

Then run:

```bash
python run.py --source video/china1.mp4 --config config/navigation.json
```

---

## Known Issues / TODO

- Currently tested only on a small number of videos, robustness in real-world scenarios needs improvement
- LLM scene descriptions depend on local Ollama service, ensure network and GPU resources are available
- Text-to-speech is based on `pyttsx3`, voice quality may vary across platforms
