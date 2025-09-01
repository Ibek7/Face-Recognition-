# YOLO Batch Inference

This project provides a simple example of running batch inference using a pre-trained YOLOv8 model from the Ultralytics library. It processes all images in a folder and saves the detection results.

## Requirements

- Python 3.8 or higher
- Ultralytics YOLOv8

## Installation

```bash
pip install ultralytics
```

## Usage

1. Place your images in a folder named `images/` in the project directory.
2. Run the script:

```bash
python yolo_batch.py
```

3. Detection results will be saved in the `runs/detect/predict` directory.

## Notes

- The script uses the default YOLOv8n (nano) model for speed and simplicity.
- You can change the model or parameters in the script as needed.
